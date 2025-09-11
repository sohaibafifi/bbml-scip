ROOT := $(abspath ..)
BUILD := $(ROOT)/build
PLUGIN_SO := $(BUILD)/cpp/libbbml_scip.so
RUNNER := $(BUILD)/bbml_run
OUT := out
MODEL ?= mlp
GRAPH ?= 0
SCORE := $(OUT)/score_$(MODEL).onnx
TIME ?= 60
NODES ?= 1000

.PHONY: build scip-log parquet train-export scip-ml clean pipeline \
        scip-log-train scip-log-val parquet-train parquet-val \
        benchmark benchmark-nobbml benchmark-compare

build:
	cmake -S $(ROOT) -B $(BUILD) -DBBML_WITH_ONNX=ON || cmake -S $(ROOT) -B $(BUILD)
	cmake --build $(BUILD) -j

scip-log: build
	BBML_TELEMETRY_PATH=$(ROOT)/examples/$(OUT)/train.ndjson \
	BBML_INSTANCE_ID=toy \
	sh -lc 'mkdir -p $(OUT) && printf "bbml/enable = TRUE\nbbml/telemetry = TRUE\nbbml/telemetry/path = \"$(ROOT)/examples/$(OUT)/train.ndjson\"\nbbml/telemetry/graph = %s\nbbml/telemetry/graph_path = \"$(ROOT)/examples/$(OUT)/train_graph.ndjson\"\nbbml/telemetry/append = FALSE\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" $$([ "$(GRAPH)" = "1" ] && echo TRUE || echo FALSE) > $(OUT)/bbml_train.set' && \
	$(RUNNER) --plugin $(PLUGIN_SO) --problem $(ROOT)/examples/data/branching.lp --set $(OUT)/bbml_train.set

parquet: scip-log
	@in=$(OUT)/train.ndjson; \
	if [ ! -s "$$in" ] && [ -s "data/logs/train.ndjson" ]; then in="data/logs/train.ndjson"; fi; \
	python -m bbml.data.json_to_parquet --in $$in --out $(OUT)/train.parquet

train-export: # parquet or graph
	@if [ "$(MODEL)" = "gnn" ] && [ "$(GRAPH)" = "1" ]; then \
	  echo "[train] GNN with graph inputs from $(OUT)/train_graph.ndjson"; \
	  python -m bbml.train.train_rank --graph_ndjson $(OUT)/train_graph.ndjson --epochs 100 --batch_size 8 --ckpt $(OUT)/model.pt --model gnn; \
	else \
	  echo "[train] $(MODEL) from parquet $(OUT)/train.parquet"; \
	  python -m bbml.train.train_rank --parquet $(OUT)/train.parquet --epochs 100 --batch_size 8 --ckpt $(OUT)/model.pt --model $(MODEL); \
	fi
	python -m bbml.export.export_onnx --out $(SCORE) --ckpt $(OUT)/model.pt --model $(MODEL)
	# Latency tool expects a single [m,d] input; works for MLP and var-only GNN
	python -m bbml.bench.latency --onnx $(SCORE) --d 6 --runs 20 || true

scip-ml: build
	BBML_TELEMETRY_PATH=$(ROOT)/examples/$(OUT)/online.ndjson \
	BBML_INSTANCE_ID=toy-ml \
	sh -lc 'mkdir -p $(OUT) && printf "bbml/enable = TRUE\nbbml/telemetry = TRUE\nbbml/model_path = $(ROOT)/examples/$(SCORE)\nbbml/confidence = 0.6\nbbml/reload = 1\nbbml/telemetry/path = \"$(ROOT)/examples/$(OUT)/online.ndjson\"\nbbml/telemetry/append = FALSE\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" > $(OUT)/bbml_online.set' && \
	$(RUNNER) --plugin $(PLUGIN_SO) --problem $(ROOT)/examples/data/branching.lp --set $(OUT)/bbml_online.set

pipeline: train-export scip-ml

# Log training split across all LPs under examples/data/train into out/train.ndjson
scip-log-train: build
	@rm -f $(OUT)/train.ndjson
	@mkdir -p $(OUT)
	@first=1; \
	for f in $(ROOT)/examples/data/train/*.lp; do \
	  echo "[train-log] $$f"; \
	  if [ $$first -eq 1 ]; then append=FALSE; else append=TRUE; fi; \
  printf "bbml/enable = TRUE\nbbml/telemetry = TRUE\nbbml/telemetry/path = \"$(ROOT)/examples/$(OUT)/train.ndjson\"\nbbml/telemetry/graph = %s\nbbml/telemetry/graph_path = \"$(ROOT)/examples/$(OUT)/train_graph.ndjson\"\nbbml/telemetry/append = %s\nbbml/telemetry/strongbranch = FALSE\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" $$([ "$(GRAPH)" = "1" ] && echo TRUE || echo FALSE) $$append > $(OUT)/bbml_train_batch.set; \
	  BBML_TELEMETRY_PATH=$(ROOT)/examples/$(OUT)/train.ndjson \
	  BBML_INSTANCE_ID=train \
	  $(RUNNER) --plugin $(PLUGIN_SO) --problem $$f --set $(OUT)/bbml_train_batch.set; \
	  first=0; \
	done

# Log validation split across all LPs under examples/data/val into out/val.ndjson
scip-log-val: build
	@rm -f $(OUT)/val.ndjson
	@mkdir -p $(OUT)
	@first=1; \
	for f in $(ROOT)/examples/data/val/*.lp; do \
	  echo "[val-log] $$f"; \
	  if [ $$first -eq 1 ]; then append=FALSE; else append=TRUE; fi; \
  printf "bbml/enable = TRUE\nbbml/telemetry = TRUE\nbbml/telemetry/path = \"$(ROOT)/examples/$(OUT)/val.ndjson\"\nbbml/telemetry/graph = %s\nbbml/telemetry/graph_path = \"$(ROOT)/examples/$(OUT)/val_graph.ndjson\"\nbbml/telemetry/append = %s\nbbml/telemetry/strongbranch = FALSE\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" $$([ "$(GRAPH)" = "1" ] && echo TRUE || echo FALSE) $$append > $(OUT)/bbml_val_batch.set; \
	  BBML_TELEMETRY_PATH=$(ROOT)/examples/$(OUT)/val.ndjson \
	  BBML_INSTANCE_ID=val \
	  $(RUNNER) --plugin $(PLUGIN_SO) --problem $$f --set $(OUT)/bbml_val_batch.set; \
	  first=0; \
	done

parquet-train: #scip-log-train
	python -m bbml.data.json_to_parquet --in $(OUT)/train.ndjson --out $(OUT)/train.parquet --chunksize 100000 --row-group-size 100000 --compression snappy

parquet-val: # scip-log-val
	python -m bbml.data.json_to_parquet --in $(OUT)/val.ndjson --out $(OUT)/val.parquet

benchmark: #train-export parquet-val
	@rm -f $(OUT)/bench_kpis.txt
	@touch $(OUT)/bench_kpis.txt
	@for f in $(ROOT)/examples/data/val/*.lp; do \
	  echo "[benchmark] $$f"; \
  printf "bbml/enable = TRUE\nbbml/telemetry = FALSE\nbbml/model_path = \"$(ROOT)/examples/$(SCORE)\"\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" > $(OUT)/bbml_bench.set; \
	  OUTSTR=`$(RUNNER) --plugin $(PLUGIN_SO) --problem $$f --set $(OUT)/bbml_bench.set`; \
	  echo "$$OUTSTR" | awk -v T=$(TIME) '/BBML_SUMMARY/ { \
	    for(i=1;i<=NF;i++){ split($$i, kv, "="); if(kv[1]=="nodes") nodes=kv[2]; else if(kv[1]=="time") time=kv[2]; else if(kv[1]=="gap") gap=kv[2]; else if(kv[1]=="primal_integral") pint=kv[2]; else if(kv[1]=="solved") solved=kv[2]; } \
	    if(nodes==""||nodes==0) nodes=0; if(time==""||time==0) time=0; if(gap=="" ) gap=1; if(pint=="" ) pint=0; if(solved=="") solved=0; \
	    par2 = (solved==1) ? time : 2*T; \
	    printf("nodes=%s time=%s gap=%s primal_integral=%s solved=%s par2=%s\n", nodes, time, gap, pint, solved, par2); \
	  }'; \
	  echo "$$OUTSTR" | awk -v T=$(TIME) '/BBML_SUMMARY/ { \
	    for(i=1;i<=NF;i++){ split($$i, kv, "="); if(kv[1]=="nodes") nodes=kv[2]; else if(kv[1]=="time") time=kv[2]; else if(kv[1]=="gap") gap=kv[2]; else if(kv[1]=="primal_integral") pint=kv[2]; else if(kv[1]=="solved") solved=kv[2]; } \
	    if(nodes==""||nodes==0) nodes=0; if(time==""||time==0) time=0; if(gap=="" ) gap=1; if(pint=="" ) pint=0; if(solved=="") solved=0; \
	    par2 = (solved==1) ? time : 2*T; \
	    printf("%s %s %s %s %s %s\n", nodes, time, gap, pint, solved, par2); \
	  }' >> $(OUT)/bench_kpis.txt; \
	done
	@awk '{nodes+=$$1; time+=$$2; gap+=$$3; pint+=$$4; solved+=$$5; par2+=$$6; c++} END{ if(c>0) printf("summary files=%d avg_nodes=%.2f avg_time=%.6f avg_gap=%.4f avg_primalint=%.4f solved=%d/%d par2=%.6f\n", c, nodes/c, time/c, gap/c, pint/c, solved, c, par2/c); else print "summary files=0" }' $(OUT)/bench_kpis.txt

# Baseline benchmark without BBML (bbml/enable = FALSE)
benchmark-nobbml: #parquet-val
	@rm -f $(OUT)/bench_kpis_nobbml.txt
	@touch $(OUT)/bench_kpis_nobbml.txt
	@for f in $(ROOT)/examples/data/val/*.lp; do \
	  echo "[baseline] $$f"; \
	  printf "bbml/enable = FALSE\nbbml/telemetry = FALSE\nlimits/time = $(TIME)\nlimits/nodes = $(NODES)\n" > $(OUT)/bbml_bench_nobbml.set; \
	  OUTSTR=`$(RUNNER) --plugin $(PLUGIN_SO) --problem $$f --set $(OUT)/bbml_bench_nobbml.set`; \
	  echo "$$OUTSTR" | awk -v T=$(TIME) '/BBML_SUMMARY/ { \
	    for(i=1;i<=NF;i++){ split($$i, kv, "="); if(kv[1]=="nodes") nodes=kv[2]; else if(kv[1]=="time") time=kv[2]; else if(kv[1]=="gap") gap=kv[2]; else if(kv[1]=="primal_integral") pint=kv[2]; else if(kv[1]=="solved") solved=kv[2]; } \
	    if(nodes==""||nodes==0) nodes=0; if(time==""||time==0) time=0; if(gap=="" ) gap=1; if(pint=="" ) pint=0; if(solved=="") solved=0; \
	    par2 = (solved==1) ? time : 2*T; \
	    printf("nodes=%s time=%s gap=%s primal_integral=%s solved=%s par2=%s\n", nodes, time, gap, pint, solved, par2); \
	  }'; \
	  echo "$$OUTSTR" | awk -v T=$(TIME) '/BBML_SUMMARY/ { \
	    for(i=1;i<=NF;i++){ split($$i, kv, "="); if(kv[1]=="nodes") nodes=kv[2]; else if(kv[1]=="time") time=kv[2]; else if(kv[1]=="gap") gap=kv[2]; else if(kv[1]=="primal_integral") pint=kv[2]; else if(kv[1]=="solved") solved=kv[2]; } \
	    if(nodes==""||nodes==0) nodes=0; if(time==""||time==0) time=0; if(gap=="" ) gap=1; if(pint=="" ) pint=0; if(solved=="") solved=0; \
	    par2 = (solved==1) ? time : 2*T; \
	    printf("%s %s %s %s %s %s\n", nodes, time, gap, pint, solved, par2); \
	  }' >> $(OUT)/bench_kpis_nobbml.txt; \
	done
	@awk '{nodes+=$$1; time+=$$2; gap+=$$3; pint+=$$4; solved+=$$5; par2+=$$6; c++} END{ if(c>0) printf("baseline files=%d avg_nodes=%.2f avg_time=%.6f avg_gap=%.4f avg_primalint=%.4f solved=%d/%d par2=%.6f\n", c, nodes/c, time/c, gap/c, pint/c, solved, c, par2/c); else print "baseline files=0" }' $(OUT)/bench_kpis_nobbml.txt

# Run both and print side-by-side summary
benchmark-compare: benchmark benchmark-nobbml
	@echo "== Comparison (BBML vs Baseline) =="
	@bb_nodes=`awk '{n+=$$1; c++} END{if(c>0) printf("%.2f", n/c); else print 0}' $(OUT)/bench_kpis.txt`; \
	  bb_time=`awk '{t+=$$2; c++} END{if(c>0) printf("%.6f", t/c); else print 0}' $(OUT)/bench_kpis.txt`; \
	  bb_gap=`awk '{g+=$$3; c++} END{if(c>0) printf("%.4f", g/c); else print 0}' $(OUT)/bench_kpis.txt`; \
	  bb_par2=`awk '{p+=$$6; c++} END{if(c>0) printf("%.6f", p/c); else print 0}' $(OUT)/bench_kpis.txt`; \
	  bl_nodes=`awk '{n+=$$1; c++} END{if(c>0) printf("%.2f", n/c); else print 0}' $(OUT)/bench_kpis_nobbml.txt`; \
	  bl_time=`awk '{t+=$$2; c++} END{if(c>0) printf("%.6f", t/c); else print 0}' $(OUT)/bench_kpis_nobbml.txt`; \
	  bl_gap=`awk '{g+=$$3; c++} END{if(c>0) printf("%.4f", g/c); else print 0}' $(OUT)/bench_kpis_nobbml.txt`; \
	  bl_par2=`awk '{p+=$$6; c++} END{if(c>0) printf("%.6f", p/c); else print 0}' $(OUT)/bench_kpis_nobbml.txt`; \
	  echo "avg_nodes: bbml=$$bb_nodes baseline=$$bl_nodes"; \
	  echo "avg_time : bbml=$$bb_time baseline=$$bl_time"; \
	  echo "avg_gap  : bbml=$$bb_gap baseline=$$bl_gap"; \
	  echo "avg_PAR2 : bbml=$$bb_par2 baseline=$$bl_par2";

clean:
	rm -rf $(OUT)
