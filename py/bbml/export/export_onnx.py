import argparse
import os
from typing import Optional, Tuple, Any, Dict

import torch
from bbml.train.train_rank import ScoreMLP
from bbml.models.graph_ranker import GraphRanker

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    _HAS_ORT_QUANT = True
except Exception:  # pragma: no cover - optional
    _HAS_ORT_QUANT = False


def _quantize_int8(in_path: str, out_path: Optional[str] = None) -> Optional[str]:
    if not _HAS_ORT_QUANT:
        print("onnxruntime.quantization not available; skipping INT8 quantization")
        return None
    out = out_path or os.path.splitext(in_path)[0] + ".int8.onnx"
    quantize_dynamic(in_path, out, weight_type=QuantType.QInt8)
    return out


def _load_ckpt(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not (path and os.path.isfile(path)):
        return None, None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj.get("state_dict"), obj.get("cfg")
    if isinstance(obj, dict):
        # Could be already a state_dict
        return obj, None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output ONNX file path")
    ap.add_argument("--model", type=str, default="mlp", choices=["mlp", "gnn"], help="Model type to export")
    ap.add_argument("--d_in", type=int, default=6, help="Input dim for MLP")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true", help="Export weights as FP16")
    ap.add_argument("--ckpt", help="Load model weights from checkpoint", type=str, default="score_mlp.pt")
    # GNN-specific
    ap.add_argument("--d_var", type=int, default=6, help="Variable feature dim for GNN")
    ap.add_argument("--d_con", type=int, default=6, help="Constraint feature dim for GNN")
    ap.add_argument("--layers", type=int, default=3, help="Number of GNN layers")
    ap.add_argument("--graph_inputs", action="store_true", help="Export GNN with graph inputs (var, con, edge_index) instead of var-only input")
    ap.add_argument(
        "--int8",
        action="store_true",
        help="Quantize dynamically to INT8 (if available)",
    )
    args = ap.parse_args()

    # Try to read model config from checkpoint if available
    state_dict, cfg = _load_ckpt(args.ckpt)
    model_type = cfg.get("model", args.model) if cfg else args.model

    if model_type == "mlp":
        d_in = cfg.get("d_in", args.d_in) if cfg else args.d_in
        hidden = cfg.get("hidden", args.hidden) if cfg else args.hidden
        dropout = cfg.get("dropout", args.dropout) if cfg else args.dropout

        model = ScoreMLP(d_in=d_in, hidden=hidden, dropout=dropout)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {args.ckpt}")
        model.eval()
        dummy = torch.randn(16, d_in)
        if args.fp16:
            model.half()
            dummy = dummy.half()
        torch.onnx.export(
            model,
            dummy,
            args.out,
            input_names=["X"],
            output_names=["scores"],
            dynamic_axes={"X": {0: "m"}, "scores": {0: "m"}},
            opset_version=17,
        )
        print(f"Exported ONNX to {args.out}")
    else:
        # GNN export
        d_var = cfg.get("d_var", args.d_var) if cfg else args.d_var
        d_con = cfg.get("d_con", args.d_con) if cfg else args.d_con
        hidden = cfg.get("hidden", args.hidden) if cfg else args.hidden
        layers = cfg.get("layers", args.layers) if cfg else args.layers
        dropout = cfg.get("dropout", args.dropout) if cfg else args.dropout
        default_graph_inputs = cfg.get("graph_inputs", False) if cfg else False
        use_graph_inputs = args.graph_inputs or default_graph_inputs

        model = GraphRanker(
            d_var=d_var,
            d_con=d_con,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {args.ckpt}")
        model.eval()

        if use_graph_inputs:
            # Export full graph signature: (var_feat, con_feat, edge_index)
            n_var, n_con, E = 16, 8, 32
            var_feat = torch.randn(n_var, d_var)
            con_feat = torch.randn(n_con, d_con)
            # Build valid edges within ranges
            rows = torch.randint(0, n_con, (E,), dtype=torch.long)
            cols = torch.randint(0, n_var, (E,), dtype=torch.long)
            edge_index = torch.stack([rows, cols], dim=0)
            if args.fp16:
                model.half()
                var_feat = var_feat.half()
                con_feat = con_feat.half()
            torch.onnx.export(
                model,
                (var_feat, con_feat, edge_index),
                args.out,
                input_names=["var_feat", "con_feat", "edge_index"],
                output_names=["scores"],
                dynamic_axes={
                    "var_feat": {0: "n_var"},
                    "con_feat": {0: "n_con"},
                    "edge_index": {1: "E"},
                    "scores": {0: "n_var"},
                },
                opset_version=17,
            )
        else:
            # Export var-only signature: (var_feat) → scores
            n_var = 16
            var_feat = torch.randn(n_var, d_var)
            if args.fp16:
                model.half()
                var_feat = var_feat.half()

            class VarOnlyWrapper(torch.nn.Module):
                def __init__(self, base: GraphRanker):
                    super().__init__()
                    self.base = base

                def forward(self, var_feat: torch.Tensor) -> torch.Tensor:  # type: ignore
                    return self.base(var_feat, None, None)

            wrapper = VarOnlyWrapper(model)
            torch.onnx.export(
                wrapper,
                var_feat,
                args.out,
                input_names=["var_feat"],
                output_names=["scores"],
                dynamic_axes={"var_feat": {0: "n_var"}, "scores": {0: "n_var"}},
                opset_version=17,
            )
        print(f"Exported ONNX to {args.out}")

    if args.int8:
        out_int8 = _quantize_int8(args.out)
        if out_int8:
            print(f"Quantized INT8 model saved to {out_int8}")


if __name__ == "__main__":
    main()
