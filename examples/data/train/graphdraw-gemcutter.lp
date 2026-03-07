\ SCIP STATISTICS
\   Problem name     : graphdraw-gemcutter
\   Variables        : 166 (112 binary, 16 integer, 0 implicit integer, 38 continuous)
\   Constraints      : 474
Minimize
 Obj: +64 dist[c,Rubygem,Dependency] +64 dist[c,Version,Dependency] +1 center_dist[r,Rubygem] +1 center_dist[r,User]
      +1 center_dist[r,WebHook] +1 center_dist[r,Subscription] +1 center_dist[r,Ownership] +1 center_dist[r,Linkset]
      +1 center_dist[r,Version] +1 center_dist[r,Dependency] +1 center_dist[c,Rubygem] +1 center_dist[c,User]
      +1 center_dist[c,WebHook] +1 center_dist[c,Subscription] +1 center_dist[c,Ownership] +1 center_dist[c,Linkset]
      +1 center_dist[c,Version] +1 center_dist[c,Dependency] +64 dist[r,Rubygem,User] +64 dist[r,User,WebHook]
      +64 dist[r,User,Subscription] +64 dist[r,User,Ownership] +64 dist[r,Rubygem,Subscription] +64 dist[r,Rubygem,WebHook]
      +64 dist[r,Rubygem,Linkset] +64 dist[r,Rubygem,Ownership] +64 dist[r,Rubygem,Version] +64 dist[r,Rubygem,Dependency]
      +64 dist[r,Version,Dependency] +64 dist[c,Rubygem,User] +64 dist[c,User,WebHook] +64 dist[c,User,Subscription]
      +64 dist[c,User,Ownership] +64 dist[c,Rubygem,Subscription] +64 dist[c,Rubygem,WebHook] +64 dist[c,Rubygem,Linkset]
      +64 dist[c,Rubygem,Ownership] +64 dist[c,Rubygem,Version]
Subject to
 choose1[Rubygem,User]: +1 zV[r,rt,Rubygem,User] +1 zV[c,rt,Rubygem,User] +1 zV[r,lb,Rubygem,User] +1 zV[c,lb,Rubygem,User]
  = +1
 choose1[Rubygem,WebHook]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[c,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,WebHook]
  +1 zV[c,lb,Rubygem,WebHook] = +1
 choose1[Rubygem,Subscription]: +1 zV[r,rt,Rubygem,Subscription] +1 zV[c,rt,Rubygem,Subscription] +1 zV[r,lb,Rubygem,Subscription]
  +1 zV[c,lb,Rubygem,Subscription] = +1
 choose1[Rubygem,Ownership]: +1 zV[r,rt,Rubygem,Ownership] +1 zV[c,rt,Rubygem,Ownership] +1 zV[r,lb,Rubygem,Ownership]
  +1 zV[c,lb,Rubygem,Ownership] = +1
 choose1[Rubygem,Linkset]: +1 zV[r,rt,Rubygem,Linkset] +1 zV[c,rt,Rubygem,Linkset] +1 zV[r,lb,Rubygem,Linkset]
  +1 zV[c,lb,Rubygem,Linkset] = +1
 choose1[Rubygem,Version]: +1 zV[r,rt,Rubygem,Version] +1 zV[c,rt,Rubygem,Version] +1 zV[r,lb,Rubygem,Version]
  +1 zV[c,lb,Rubygem,Version] = +1
 choose1[Rubygem,Dependency]: +1 zV[r,rt,Rubygem,Dependency] +1 zV[c,rt,Rubygem,Dependency] +1 zV[r,lb,Rubygem,Dependency]
  +1 zV[c,lb,Rubygem,Dependency] = +1
 choose1[User,WebHook]: +1 zV[r,rt,User,WebHook] +1 zV[c,rt,User,WebHook] +1 zV[r,lb,User,WebHook] +1 zV[c,lb,User,WebHook]
  = +1
 choose1[User,Subscription]: +1 zV[r,rt,User,Subscription] +1 zV[c,rt,User,Subscription] +1 zV[r,lb,User,Subscription]
  +1 zV[c,lb,User,Subscription] = +1
 choose1[User,Ownership]: +1 zV[r,rt,User,Ownership] +1 zV[c,rt,User,Ownership] +1 zV[r,lb,User,Ownership]
  +1 zV[c,lb,User,Ownership] = +1
 choose1[User,Linkset]: +1 zV[r,rt,User,Linkset] +1 zV[c,rt,User,Linkset] +1 zV[r,lb,User,Linkset] +1 zV[c,lb,User,Linkset]
  = +1
 choose1[User,Version]: +1 zV[r,rt,User,Version] +1 zV[c,rt,User,Version] +1 zV[r,lb,User,Version] +1 zV[c,lb,User,Version]
  = +1
 choose1[User,Dependency]: +1 zV[r,rt,User,Dependency] +1 zV[c,rt,User,Dependency] +1 zV[r,lb,User,Dependency]
  +1 zV[c,lb,User,Dependency] = +1
 choose1[WebHook,Subscription]: +1 zV[r,rt,WebHook,Subscription] +1 zV[c,rt,WebHook,Subscription] +1 zV[r,lb,WebHook,Subscription]
  +1 zV[c,lb,WebHook,Subscription] = +1
 choose1[WebHook,Ownership]: +1 zV[r,rt,WebHook,Ownership] +1 zV[c,rt,WebHook,Ownership] +1 zV[r,lb,WebHook,Ownership]
  +1 zV[c,lb,WebHook,Ownership] = +1
 choose1[WebHook,Linkset]: +1 zV[r,rt,WebHook,Linkset] +1 zV[c,rt,WebHook,Linkset] +1 zV[r,lb,WebHook,Linkset]
  +1 zV[c,lb,WebHook,Linkset] = +1
 choose1[WebHook,Version]: +1 zV[r,rt,WebHook,Version] +1 zV[c,rt,WebHook,Version] +1 zV[r,lb,WebHook,Version]
  +1 zV[c,lb,WebHook,Version] = +1
 choose1[WebHook,Dependency]: +1 zV[r,rt,WebHook,Dependency] +1 zV[c,rt,WebHook,Dependency] +1 zV[r,lb,WebHook,Dependency]
  +1 zV[c,lb,WebHook,Dependency] = +1
 choose1[Subscription,Ownership]: +1 zV[r,rt,Subscription,Ownership] +1 zV[c,rt,Subscription,Ownership]
  +1 zV[r,lb,Subscription,Ownership] +1 zV[c,lb,Subscription,Ownership] = +1
 choose1[Subscription,Linkset]: +1 zV[r,rt,Subscription,Linkset] +1 zV[c,rt,Subscription,Linkset] +1 zV[r,lb,Subscription,Linkset]
  +1 zV[c,lb,Subscription,Linkset] = +1
 choose1[Subscription,Version]: +1 zV[r,rt,Subscription,Version] +1 zV[c,rt,Subscription,Version] +1 zV[r,lb,Subscription,Version]
  +1 zV[c,lb,Subscription,Version] = +1
 choose1[Subscription,Dependency]: +1 zV[r,rt,Subscription,Dependency] +1 zV[c,rt,Subscription,Dependency]
  +1 zV[r,lb,Subscription,Dependency] +1 zV[c,lb,Subscription,Dependency] = +1
 choose1[Ownership,Linkset]: +1 zV[r,rt,Ownership,Linkset] +1 zV[c,rt,Ownership,Linkset] +1 zV[r,lb,Ownership,Linkset]
  +1 zV[c,lb,Ownership,Linkset] = +1
 choose1[Ownership,Version]: +1 zV[r,rt,Ownership,Version] +1 zV[c,rt,Ownership,Version] +1 zV[r,lb,Ownership,Version]
  +1 zV[c,lb,Ownership,Version] = +1
 choose1[Ownership,Dependency]: +1 zV[r,rt,Ownership,Dependency] +1 zV[c,rt,Ownership,Dependency] +1 zV[r,lb,Ownership,Dependency]
  +1 zV[c,lb,Ownership,Dependency] = +1
 choose1[Linkset,Version]: +1 zV[r,rt,Linkset,Version] +1 zV[c,rt,Linkset,Version] +1 zV[r,lb,Linkset,Version]
  +1 zV[c,lb,Linkset,Version] = +1
 choose1[Linkset,Dependency]: +1 zV[r,rt,Linkset,Dependency] +1 zV[c,rt,Linkset,Dependency] +1 zV[r,lb,Linkset,Dependency]
  +1 zV[c,lb,Linkset,Dependency] = +1
 choose1[Version,Dependency]: +1 zV[r,rt,Version,Dependency] +1 zV[c,rt,Version,Dependency] +1 zV[r,lb,Version,Dependency]
  +1 zV[c,lb,Version,Dependency] = +1
 centerDistAxis1[r,Rubygem]: +1 x[r,Rubygem] +1 center_dist[r,Rubygem] >= +14
 centerDistAxis1[r,User]: +1 x[r,User] +1 center_dist[r,User] >= +12
 centerDistAxis1[r,WebHook]: +1 x[r,WebHook] +1 center_dist[r,WebHook] >= +14
 centerDistAxis1[r,Subscription]: +1 x[r,Subscription] +1 center_dist[r,Subscription] >= +14.5
 centerDistAxis1[r,Ownership]: +1 x[r,Ownership] +1 center_dist[r,Ownership] >= +14
 centerDistAxis1[r,Linkset]: +1 x[r,Linkset] +1 center_dist[r,Linkset] >= +13
 centerDistAxis1[r,Version]: +1 x[r,Version] +1 center_dist[r,Version] >= +12
 centerDistAxis1[r,Dependency]: +1 x[r,Dependency] +1 center_dist[r,Dependency] >= +14
 centerDistAxis1[c,Rubygem]: +1 x[c,Rubygem] +1 center_dist[c,Rubygem] >= +12
 centerDistAxis1[c,User]: +1 x[c,User] +1 center_dist[c,User] >= +9.5
 centerDistAxis1[c,WebHook]: +1 x[c,WebHook] +1 center_dist[c,WebHook] >= +11
 centerDistAxis1[c,Subscription]: +1 x[c,Subscription] +1 center_dist[c,Subscription] >= +12.5
 centerDistAxis1[c,Ownership]: +1 x[c,Ownership] +1 center_dist[c,Ownership] >= +12
 centerDistAxis1[c,Linkset]: +1 x[c,Linkset] +1 center_dist[c,Linkset] >= +13
 centerDistAxis1[c,Version]: +1 x[c,Version] +1 center_dist[c,Version] >= +10.5
 centerDistAxis1[c,Dependency]: +1 x[c,Dependency] +1 center_dist[c,Dependency] >= +11.5
 centerDistAxis2[r,Rubygem]: -1 x[r,Rubygem] +1 center_dist[r,Rubygem] >= -14
 centerDistAxis2[r,User]: -1 x[r,User] +1 center_dist[r,User] >= -12
 centerDistAxis2[r,WebHook]: -1 x[r,WebHook] +1 center_dist[r,WebHook] >= -14
 centerDistAxis2[r,Subscription]: -1 x[r,Subscription] +1 center_dist[r,Subscription] >= -14.5
 centerDistAxis2[r,Ownership]: -1 x[r,Ownership] +1 center_dist[r,Ownership] >= -14
 centerDistAxis2[r,Linkset]: -1 x[r,Linkset] +1 center_dist[r,Linkset] >= -13
 centerDistAxis2[r,Version]: -1 x[r,Version] +1 center_dist[r,Version] >= -12
 centerDistAxis2[r,Dependency]: -1 x[r,Dependency] +1 center_dist[r,Dependency] >= -14
 centerDistAxis2[c,Rubygem]: -1 x[c,Rubygem] +1 center_dist[c,Rubygem] >= -12
 centerDistAxis2[c,User]: -1 x[c,User] +1 center_dist[c,User] >= -9.5
 centerDistAxis2[c,WebHook]: -1 x[c,WebHook] +1 center_dist[c,WebHook] >= -11
 centerDistAxis2[c,Subscription]: -1 x[c,Subscription] +1 center_dist[c,Subscription] >= -12.5
 centerDistAxis2[c,Ownership]: -1 x[c,Ownership] +1 center_dist[c,Ownership] >= -12
 centerDistAxis2[c,Linkset]: -1 x[c,Linkset] +1 center_dist[c,Linkset] >= -13
 centerDistAxis2[c,Version]: -1 x[c,Version] +1 center_dist[c,Version] >= -10.5
 centerDistAxis2[c,Dependency]: -1 x[c,Dependency] +1 center_dist[c,Dependency] >= -11.5
 axisVdifLB[r,Rubygem,User]: -1 x[r,Rubygem] +1 x[r,User] -36 zV[r,lb,Rubygem,User] >= -31
 axisVdifLB[r,Rubygem,WebHook]: -1 x[r,Rubygem] +1 x[r,WebHook] -36 zV[r,lb,Rubygem,WebHook] >= -31
 axisVdifLB[r,Rubygem,Subscription]: -1 x[r,Rubygem] +1 x[r,Subscription] -36 zV[r,lb,Rubygem,Subscription]
  >= -31
 axisVdifLB[r,Rubygem,Ownership]: -1 x[r,Rubygem] +1 x[r,Ownership] -36 zV[r,lb,Rubygem,Ownership] >= -31
 axisVdifLB[r,Rubygem,Linkset]: -1 x[r,Rubygem] +1 x[r,Linkset] -36 zV[r,lb,Rubygem,Linkset] >= -31
 axisVdifLB[r,Rubygem,Version]: -1 x[r,Rubygem] +1 x[r,Version] -36 zV[r,lb,Rubygem,Version] >= -31
 axisVdifLB[r,Rubygem,Dependency]: -1 x[r,Rubygem] +1 x[r,Dependency] -36 zV[r,lb,Rubygem,Dependency]
  >= -31
 axisVdifLB[r,User,WebHook]: -1 x[r,User] +1 x[r,WebHook] -40 zV[r,lb,User,WebHook] >= -31
 axisVdifLB[r,User,Subscription]: -1 x[r,User] +1 x[r,Subscription] -40 zV[r,lb,User,Subscription] >= -31
 axisVdifLB[r,User,Ownership]: -1 x[r,User] +1 x[r,Ownership] -40 zV[r,lb,User,Ownership] >= -31
 axisVdifLB[r,User,Linkset]: -1 x[r,User] +1 x[r,Linkset] -40 zV[r,lb,User,Linkset] >= -31
 axisVdifLB[r,User,Version]: -1 x[r,User] +1 x[r,Version] -40 zV[r,lb,User,Version] >= -31
 axisVdifLB[r,User,Dependency]: -1 x[r,User] +1 x[r,Dependency] -40 zV[r,lb,User,Dependency] >= -31
 axisVdifLB[r,WebHook,Subscription]: -1 x[r,WebHook] +1 x[r,Subscription] -36 zV[r,lb,WebHook,Subscription]
  >= -31
 axisVdifLB[r,WebHook,Ownership]: -1 x[r,WebHook] +1 x[r,Ownership] -36 zV[r,lb,WebHook,Ownership] >= -31
 axisVdifLB[r,WebHook,Linkset]: -1 x[r,WebHook] +1 x[r,Linkset] -36 zV[r,lb,WebHook,Linkset] >= -31
 axisVdifLB[r,WebHook,Version]: -1 x[r,WebHook] +1 x[r,Version] -36 zV[r,lb,WebHook,Version] >= -31
 axisVdifLB[r,WebHook,Dependency]: -1 x[r,WebHook] +1 x[r,Dependency] -36 zV[r,lb,WebHook,Dependency]
  >= -31
 axisVdifLB[r,Subscription,Ownership]: -1 x[r,Subscription] +1 x[r,Ownership] -35 zV[r,lb,Subscription,Ownership]
  >= -31
 axisVdifLB[r,Subscription,Linkset]: -1 x[r,Subscription] +1 x[r,Linkset] -35 zV[r,lb,Subscription,Linkset]
  >= -31
 axisVdifLB[r,Subscription,Version]: -1 x[r,Subscription] +1 x[r,Version] -35 zV[r,lb,Subscription,Version]
  >= -31
 axisVdifLB[r,Subscription,Dependency]: -1 x[r,Subscription] +1 x[r,Dependency] -35 zV[r,lb,Subscription,Dependency]
  >= -31
 axisVdifLB[r,Ownership,Linkset]: -1 x[r,Ownership] +1 x[r,Linkset] -36 zV[r,lb,Ownership,Linkset] >= -31
 axisVdifLB[r,Ownership,Version]: -1 x[r,Ownership] +1 x[r,Version] -36 zV[r,lb,Ownership,Version] >= -31
 axisVdifLB[r,Ownership,Dependency]: -1 x[r,Ownership] +1 x[r,Dependency] -36 zV[r,lb,Ownership,Dependency]
  >= -31
 axisVdifLB[r,Linkset,Version]: -1 x[r,Linkset] +1 x[r,Version] -38 zV[r,lb,Linkset,Version] >= -31
 axisVdifLB[r,Linkset,Dependency]: -1 x[r,Linkset] +1 x[r,Dependency] -38 zV[r,lb,Linkset,Dependency]
  >= -31
 axisVdifLB[r,Version,Dependency]: -1 x[r,Version] +1 x[r,Dependency] -40 zV[r,lb,Version,Dependency]
  >= -31
 axisVdifLB[c,Rubygem,User]: -1 x[c,Rubygem] +1 x[c,User] -40 zV[c,lb,Rubygem,User] >= -31
 axisVdifLB[c,Rubygem,WebHook]: -1 x[c,Rubygem] +1 x[c,WebHook] -40 zV[c,lb,Rubygem,WebHook] >= -31
 axisVdifLB[c,Rubygem,Subscription]: -1 x[c,Rubygem] +1 x[c,Subscription] -40 zV[c,lb,Rubygem,Subscription]
  >= -31
 axisVdifLB[c,Rubygem,Ownership]: -1 x[c,Rubygem] +1 x[c,Ownership] -40 zV[c,lb,Rubygem,Ownership] >= -31
 axisVdifLB[c,Rubygem,Linkset]: -1 x[c,Rubygem] +1 x[c,Linkset] -40 zV[c,lb,Rubygem,Linkset] >= -31
 axisVdifLB[c,Rubygem,Version]: -1 x[c,Rubygem] +1 x[c,Version] -40 zV[c,lb,Rubygem,Version] >= -31
 axisVdifLB[c,Rubygem,Dependency]: -1 x[c,Rubygem] +1 x[c,Dependency] -40 zV[c,lb,Rubygem,Dependency]
  >= -31
 axisVdifLB[c,User,WebHook]: -1 x[c,User] +1 x[c,WebHook] -45 zV[c,lb,User,WebHook] >= -31
 axisVdifLB[c,User,Subscription]: -1 x[c,User] +1 x[c,Subscription] -45 zV[c,lb,User,Subscription] >= -31
 axisVdifLB[c,User,Ownership]: -1 x[c,User] +1 x[c,Ownership] -45 zV[c,lb,User,Ownership] >= -31
 axisVdifLB[c,User,Linkset]: -1 x[c,User] +1 x[c,Linkset] -45 zV[c,lb,User,Linkset] >= -31
 axisVdifLB[c,User,Version]: -1 x[c,User] +1 x[c,Version] -45 zV[c,lb,User,Version] >= -31
 axisVdifLB[c,User,Dependency]: -1 x[c,User] +1 x[c,Dependency] -45 zV[c,lb,User,Dependency] >= -31
 axisVdifLB[c,WebHook,Subscription]: -1 x[c,WebHook] +1 x[c,Subscription] -42 zV[c,lb,WebHook,Subscription]
  >= -31
 axisVdifLB[c,WebHook,Ownership]: -1 x[c,WebHook] +1 x[c,Ownership] -42 zV[c,lb,WebHook,Ownership] >= -31
 axisVdifLB[c,WebHook,Linkset]: -1 x[c,WebHook] +1 x[c,Linkset] -42 zV[c,lb,WebHook,Linkset] >= -31
 axisVdifLB[c,WebHook,Version]: -1 x[c,WebHook] +1 x[c,Version] -42 zV[c,lb,WebHook,Version] >= -31
 axisVdifLB[c,WebHook,Dependency]: -1 x[c,WebHook] +1 x[c,Dependency] -42 zV[c,lb,WebHook,Dependency]
  >= -31
 axisVdifLB[c,Subscription,Ownership]: -1 x[c,Subscription] +1 x[c,Ownership] -39 zV[c,lb,Subscription,Ownership]
  >= -31
 axisVdifLB[c,Subscription,Linkset]: -1 x[c,Subscription] +1 x[c,Linkset] -39 zV[c,lb,Subscription,Linkset]
  >= -31
 axisVdifLB[c,Subscription,Version]: -1 x[c,Subscription] +1 x[c,Version] -39 zV[c,lb,Subscription,Version]
  >= -31
 axisVdifLB[c,Subscription,Dependency]: -1 x[c,Subscription] +1 x[c,Dependency] -39 zV[c,lb,Subscription,Dependency]
  >= -31
 axisVdifLB[c,Ownership,Linkset]: -1 x[c,Ownership] +1 x[c,Linkset] -40 zV[c,lb,Ownership,Linkset] >= -31
 axisVdifLB[c,Ownership,Version]: -1 x[c,Ownership] +1 x[c,Version] -40 zV[c,lb,Ownership,Version] >= -31
 axisVdifLB[c,Ownership,Dependency]: -1 x[c,Ownership] +1 x[c,Dependency] -40 zV[c,lb,Ownership,Dependency]
  >= -31
 axisVdifLB[c,Linkset,Version]: -1 x[c,Linkset] +1 x[c,Version] -38 zV[c,lb,Linkset,Version] >= -31
 axisVdifLB[c,Linkset,Dependency]: -1 x[c,Linkset] +1 x[c,Dependency] -38 zV[c,lb,Linkset,Dependency]
  >= -31
 axisVdifLB[c,Version,Dependency]: -1 x[c,Version] +1 x[c,Dependency] -43 zV[c,lb,Version,Dependency]
  >= -31
 axisVdifRT[r,Rubygem,User]: +1 x[r,Rubygem] -1 x[r,User] -40 zV[r,rt,Rubygem,User] >= -31
 axisVdifRT[r,Rubygem,WebHook]: +1 x[r,Rubygem] -1 x[r,WebHook] -36 zV[r,rt,Rubygem,WebHook] >= -31
 axisVdifRT[r,Rubygem,Subscription]: +1 x[r,Rubygem] -1 x[r,Subscription] -35 zV[r,rt,Rubygem,Subscription]
  >= -31
 axisVdifRT[r,Rubygem,Ownership]: +1 x[r,Rubygem] -1 x[r,Ownership] -36 zV[r,rt,Rubygem,Ownership] >= -31
 axisVdifRT[r,Rubygem,Linkset]: +1 x[r,Rubygem] -1 x[r,Linkset] -38 zV[r,rt,Rubygem,Linkset] >= -31
 axisVdifRT[r,Rubygem,Version]: +1 x[r,Rubygem] -1 x[r,Version] -40 zV[r,rt,Rubygem,Version] >= -31
 axisVdifRT[r,Rubygem,Dependency]: +1 x[r,Rubygem] -1 x[r,Dependency] -36 zV[r,rt,Rubygem,Dependency]
  >= -31
 axisVdifRT[r,User,WebHook]: +1 x[r,User] -1 x[r,WebHook] -36 zV[r,rt,User,WebHook] >= -31
 axisVdifRT[r,User,Subscription]: +1 x[r,User] -1 x[r,Subscription] -35 zV[r,rt,User,Subscription] >= -31
 axisVdifRT[r,User,Ownership]: +1 x[r,User] -1 x[r,Ownership] -36 zV[r,rt,User,Ownership] >= -31
 axisVdifRT[r,User,Linkset]: +1 x[r,User] -1 x[r,Linkset] -38 zV[r,rt,User,Linkset] >= -31
 axisVdifRT[r,User,Version]: +1 x[r,User] -1 x[r,Version] -40 zV[r,rt,User,Version] >= -31
 axisVdifRT[r,User,Dependency]: +1 x[r,User] -1 x[r,Dependency] -36 zV[r,rt,User,Dependency] >= -31
 axisVdifRT[r,WebHook,Subscription]: +1 x[r,WebHook] -1 x[r,Subscription] -35 zV[r,rt,WebHook,Subscription]
  >= -31
 axisVdifRT[r,WebHook,Ownership]: +1 x[r,WebHook] -1 x[r,Ownership] -36 zV[r,rt,WebHook,Ownership] >= -31
 axisVdifRT[r,WebHook,Linkset]: +1 x[r,WebHook] -1 x[r,Linkset] -38 zV[r,rt,WebHook,Linkset] >= -31
 axisVdifRT[r,WebHook,Version]: +1 x[r,WebHook] -1 x[r,Version] -40 zV[r,rt,WebHook,Version] >= -31
 axisVdifRT[r,WebHook,Dependency]: +1 x[r,WebHook] -1 x[r,Dependency] -36 zV[r,rt,WebHook,Dependency]
  >= -31
 axisVdifRT[r,Subscription,Ownership]: +1 x[r,Subscription] -1 x[r,Ownership] -36 zV[r,rt,Subscription,Ownership]
  >= -31
 axisVdifRT[r,Subscription,Linkset]: +1 x[r,Subscription] -1 x[r,Linkset] -38 zV[r,rt,Subscription,Linkset]
  >= -31
 axisVdifRT[r,Subscription,Version]: +1 x[r,Subscription] -1 x[r,Version] -40 zV[r,rt,Subscription,Version]
  >= -31
 axisVdifRT[r,Subscription,Dependency]: +1 x[r,Subscription] -1 x[r,Dependency] -36 zV[r,rt,Subscription,Dependency]
  >= -31
 axisVdifRT[r,Ownership,Linkset]: +1 x[r,Ownership] -1 x[r,Linkset] -38 zV[r,rt,Ownership,Linkset] >= -31
 axisVdifRT[r,Ownership,Version]: +1 x[r,Ownership] -1 x[r,Version] -40 zV[r,rt,Ownership,Version] >= -31
 axisVdifRT[r,Ownership,Dependency]: +1 x[r,Ownership] -1 x[r,Dependency] -36 zV[r,rt,Ownership,Dependency]
  >= -31
 axisVdifRT[r,Linkset,Version]: +1 x[r,Linkset] -1 x[r,Version] -40 zV[r,rt,Linkset,Version] >= -31
 axisVdifRT[r,Linkset,Dependency]: +1 x[r,Linkset] -1 x[r,Dependency] -36 zV[r,rt,Linkset,Dependency]
  >= -31
 axisVdifRT[r,Version,Dependency]: +1 x[r,Version] -1 x[r,Dependency] -36 zV[r,rt,Version,Dependency]
  >= -31
 axisVdifRT[c,Rubygem,User]: +1 x[c,Rubygem] -1 x[c,User] -45 zV[c,rt,Rubygem,User] >= -31
 axisVdifRT[c,Rubygem,WebHook]: +1 x[c,Rubygem] -1 x[c,WebHook] -42 zV[c,rt,Rubygem,WebHook] >= -31
 axisVdifRT[c,Rubygem,Subscription]: +1 x[c,Rubygem] -1 x[c,Subscription] -39 zV[c,rt,Rubygem,Subscription]
  >= -31
 axisVdifRT[c,Rubygem,Ownership]: +1 x[c,Rubygem] -1 x[c,Ownership] -40 zV[c,rt,Rubygem,Ownership] >= -31
 axisVdifRT[c,Rubygem,Linkset]: +1 x[c,Rubygem] -1 x[c,Linkset] -38 zV[c,rt,Rubygem,Linkset] >= -31
 axisVdifRT[c,Rubygem,Version]: +1 x[c,Rubygem] -1 x[c,Version] -43 zV[c,rt,Rubygem,Version] >= -31
 axisVdifRT[c,Rubygem,Dependency]: +1 x[c,Rubygem] -1 x[c,Dependency] -41 zV[c,rt,Rubygem,Dependency]
  >= -31
 axisVdifRT[c,User,WebHook]: +1 x[c,User] -1 x[c,WebHook] -42 zV[c,rt,User,WebHook] >= -31
 axisVdifRT[c,User,Subscription]: +1 x[c,User] -1 x[c,Subscription] -39 zV[c,rt,User,Subscription] >= -31
 axisVdifRT[c,User,Ownership]: +1 x[c,User] -1 x[c,Ownership] -40 zV[c,rt,User,Ownership] >= -31
 axisVdifRT[c,User,Linkset]: +1 x[c,User] -1 x[c,Linkset] -38 zV[c,rt,User,Linkset] >= -31
 axisVdifRT[c,User,Version]: +1 x[c,User] -1 x[c,Version] -43 zV[c,rt,User,Version] >= -31
 axisVdifRT[c,User,Dependency]: +1 x[c,User] -1 x[c,Dependency] -41 zV[c,rt,User,Dependency] >= -31
 axisVdifRT[c,WebHook,Subscription]: +1 x[c,WebHook] -1 x[c,Subscription] -39 zV[c,rt,WebHook,Subscription]
  >= -31
 axisVdifRT[c,WebHook,Ownership]: +1 x[c,WebHook] -1 x[c,Ownership] -40 zV[c,rt,WebHook,Ownership] >= -31
 axisVdifRT[c,WebHook,Linkset]: +1 x[c,WebHook] -1 x[c,Linkset] -38 zV[c,rt,WebHook,Linkset] >= -31
 axisVdifRT[c,WebHook,Version]: +1 x[c,WebHook] -1 x[c,Version] -43 zV[c,rt,WebHook,Version] >= -31
 axisVdifRT[c,WebHook,Dependency]: +1 x[c,WebHook] -1 x[c,Dependency] -41 zV[c,rt,WebHook,Dependency]
  >= -31
 axisVdifRT[c,Subscription,Ownership]: +1 x[c,Subscription] -1 x[c,Ownership] -40 zV[c,rt,Subscription,Ownership]
  >= -31
 axisVdifRT[c,Subscription,Linkset]: +1 x[c,Subscription] -1 x[c,Linkset] -38 zV[c,rt,Subscription,Linkset]
  >= -31
 axisVdifRT[c,Subscription,Version]: +1 x[c,Subscription] -1 x[c,Version] -43 zV[c,rt,Subscription,Version]
  >= -31
 axisVdifRT[c,Subscription,Dependency]: +1 x[c,Subscription] -1 x[c,Dependency] -41 zV[c,rt,Subscription,Dependency]
  >= -31
 axisVdifRT[c,Ownership,Linkset]: +1 x[c,Ownership] -1 x[c,Linkset] -38 zV[c,rt,Ownership,Linkset] >= -31
 axisVdifRT[c,Ownership,Version]: +1 x[c,Ownership] -1 x[c,Version] -43 zV[c,rt,Ownership,Version] >= -31
 axisVdifRT[c,Ownership,Dependency]: +1 x[c,Ownership] -1 x[c,Dependency] -41 zV[c,rt,Ownership,Dependency]
  >= -31
 axisVdifRT[c,Linkset,Version]: +1 x[c,Linkset] -1 x[c,Version] -43 zV[c,rt,Linkset,Version] >= -31
 axisVdifRT[c,Linkset,Dependency]: +1 x[c,Linkset] -1 x[c,Dependency] -41 zV[c,rt,Linkset,Dependency]
  >= -31
 axisVdifRT[c,Version,Dependency]: +1 x[c,Version] -1 x[c,Dependency] -41 zV[c,rt,Version,Dependency]
  >= -31
 compDistAxis1[r,Rubygem,User]: -1 x[r,Rubygem] +1 x[r,User] +1 dist[r,Rubygem,User] >= -2
 compDistAxis1[r,User,WebHook]: -1 x[r,User] +1 x[r,WebHook] +1 dist[r,User,WebHook] >= +2
 compDistAxis1[r,User,Subscription]: -1 x[r,User] +1 x[r,Subscription] +1 dist[r,User,Subscription] >= +2.5
 compDistAxis1[r,User,Ownership]: -1 x[r,User] +1 x[r,Ownership] +1 dist[r,User,Ownership] >= +2
 compDistAxis1[r,Rubygem,Subscription]: -1 x[r,Rubygem] +1 x[r,Subscription] +1 dist[r,Rubygem,Subscription]
  >= +0.5
 compDistAxis1[r,Rubygem,WebHook]: -1 x[r,Rubygem] +1 x[r,WebHook] +1 dist[r,Rubygem,WebHook] >= +0
 compDistAxis1[r,Rubygem,Linkset]: -1 x[r,Rubygem] +1 x[r,Linkset] +1 dist[r,Rubygem,Linkset] >= -1
 compDistAxis1[r,Rubygem,Ownership]: -1 x[r,Rubygem] +1 x[r,Ownership] +1 dist[r,Rubygem,Ownership] >= +0
 compDistAxis1[r,Rubygem,Version]: -1 x[r,Rubygem] +1 x[r,Version] +1 dist[r,Rubygem,Version] >= -2
 compDistAxis1[r,Rubygem,Dependency]: -1 x[r,Rubygem] +1 x[r,Dependency] +1 dist[r,Rubygem,Dependency]
  >= +0
 compDistAxis1[r,Version,Dependency]: -1 x[r,Version] +1 x[r,Dependency] +1 dist[r,Version,Dependency]
  >= +2
 compDistAxis1[c,Rubygem,User]: -1 x[c,Rubygem] +1 x[c,User] +1 dist[c,Rubygem,User] >= -2.5
 compDistAxis1[c,User,WebHook]: -1 x[c,User] +1 x[c,WebHook] +1 dist[c,User,WebHook] >= +1.5
 compDistAxis1[c,User,Subscription]: -1 x[c,User] +1 x[c,Subscription] +1 dist[c,User,Subscription] >= +3
 compDistAxis1[c,User,Ownership]: -1 x[c,User] +1 x[c,Ownership] +1 dist[c,User,Ownership] >= +2.5
 compDistAxis1[c,Rubygem,Subscription]: -1 x[c,Rubygem] +1 x[c,Subscription] +1 dist[c,Rubygem,Subscription]
  >= +0.5
 compDistAxis1[c,Rubygem,WebHook]: -1 x[c,Rubygem] +1 x[c,WebHook] +1 dist[c,Rubygem,WebHook] >= -1
 compDistAxis1[c,Rubygem,Linkset]: -1 x[c,Rubygem] +1 x[c,Linkset] +1 dist[c,Rubygem,Linkset] >= +1
 compDistAxis1[c,Rubygem,Ownership]: -1 x[c,Rubygem] +1 x[c,Ownership] +1 dist[c,Rubygem,Ownership] >= +0
 compDistAxis1[c,Rubygem,Version]: -1 x[c,Rubygem] +1 x[c,Version] +1 dist[c,Rubygem,Version] >= -1.5
 compDistAxis1[c,Rubygem,Dependency]: -1 x[c,Rubygem] +1 x[c,Dependency] +1 dist[c,Rubygem,Dependency]
  >= -0.5
 compDistAxis1[c,Version,Dependency]: -1 x[c,Version] +1 x[c,Dependency] +1 dist[c,Version,Dependency]
  >= +1
 compDistAxis2[r,Rubygem,User]: +1 x[r,Rubygem] -1 x[r,User] +1 dist[r,Rubygem,User] >= +2
 compDistAxis2[r,User,WebHook]: +1 x[r,User] -1 x[r,WebHook] +1 dist[r,User,WebHook] >= -2
 compDistAxis2[r,User,Subscription]: +1 x[r,User] -1 x[r,Subscription] +1 dist[r,User,Subscription] >= -2.5
 compDistAxis2[r,User,Ownership]: +1 x[r,User] -1 x[r,Ownership] +1 dist[r,User,Ownership] >= -2
 compDistAxis2[r,Rubygem,Subscription]: +1 x[r,Rubygem] -1 x[r,Subscription] +1 dist[r,Rubygem,Subscription]
  >= -0.5
 compDistAxis2[r,Rubygem,WebHook]: +1 x[r,Rubygem] -1 x[r,WebHook] +1 dist[r,Rubygem,WebHook] >= +0
 compDistAxis2[r,Rubygem,Linkset]: +1 x[r,Rubygem] -1 x[r,Linkset] +1 dist[r,Rubygem,Linkset] >= +1
 compDistAxis2[r,Rubygem,Ownership]: +1 x[r,Rubygem] -1 x[r,Ownership] +1 dist[r,Rubygem,Ownership] >= +0
 compDistAxis2[r,Rubygem,Version]: +1 x[r,Rubygem] -1 x[r,Version] +1 dist[r,Rubygem,Version] >= +2
 compDistAxis2[r,Rubygem,Dependency]: +1 x[r,Rubygem] -1 x[r,Dependency] +1 dist[r,Rubygem,Dependency]
  >= +0
 compDistAxis2[r,Version,Dependency]: +1 x[r,Version] -1 x[r,Dependency] +1 dist[r,Version,Dependency]
  >= -2
 compDistAxis2[c,Rubygem,User]: +1 x[c,Rubygem] -1 x[c,User] +1 dist[c,Rubygem,User] >= +2.5
 compDistAxis2[c,User,WebHook]: +1 x[c,User] -1 x[c,WebHook] +1 dist[c,User,WebHook] >= -1.5
 compDistAxis2[c,User,Subscription]: +1 x[c,User] -1 x[c,Subscription] +1 dist[c,User,Subscription] >= -3
 compDistAxis2[c,User,Ownership]: +1 x[c,User] -1 x[c,Ownership] +1 dist[c,User,Ownership] >= -2.5
 compDistAxis2[c,Rubygem,Subscription]: +1 x[c,Rubygem] -1 x[c,Subscription] +1 dist[c,Rubygem,Subscription]
  >= -0.5
 compDistAxis2[c,Rubygem,WebHook]: +1 x[c,Rubygem] -1 x[c,WebHook] +1 dist[c,Rubygem,WebHook] >= +1
 compDistAxis2[c,Rubygem,Linkset]: +1 x[c,Rubygem] -1 x[c,Linkset] +1 dist[c,Rubygem,Linkset] >= -1
 compDistAxis2[c,Rubygem,Ownership]: +1 x[c,Rubygem] -1 x[c,Ownership] +1 dist[c,Rubygem,Ownership] >= +0
 compDistAxis2[c,Rubygem,Version]: +1 x[c,Rubygem] -1 x[c,Version] +1 dist[c,Rubygem,Version] >= +1.5
 compDistAxis2[c,Rubygem,Dependency]: +1 x[c,Rubygem] -1 x[c,Dependency] +1 dist[c,Rubygem,Dependency]
  >= +0.5
 compDistAxis2[c,Version,Dependency]: +1 x[c,Version] -1 x[c,Dependency] +1 dist[c,Version,Dependency]
  >= -1
 DistSumLB[Rubygem,User]: +1 dist[r,Rubygem,User] +1 dist[c,Rubygem,User] >= +7
 DistSumLB[User,WebHook]: +1 dist[r,User,WebHook] +1 dist[c,User,WebHook] >= +7
 DistSumLB[User,Subscription]: +1 dist[r,User,Subscription] +1 dist[c,User,Subscription] >= +6.5
 DistSumLB[User,Ownership]: +1 dist[r,User,Ownership] +1 dist[c,User,Ownership] >= +7
 DistSumLB[Rubygem,Subscription]: +1 dist[r,Rubygem,Subscription] +1 dist[c,Rubygem,Subscription] >= +4.5
 DistSumLB[Rubygem,WebHook]: +1 dist[r,Rubygem,WebHook] +1 dist[c,Rubygem,WebHook] >= +5
 DistSumLB[Rubygem,Linkset]: +1 dist[r,Rubygem,Linkset] +1 dist[c,Rubygem,Linkset] >= +6
 DistSumLB[Rubygem,Ownership]: +1 dist[r,Rubygem,Ownership] +1 dist[c,Rubygem,Ownership] >= +5
 DistSumLB[Rubygem,Version]: +1 dist[r,Rubygem,Version] +1 dist[c,Rubygem,Version] >= +7
 DistSumLB[Rubygem,Dependency]: +1 dist[r,Rubygem,Dependency] +1 dist[c,Rubygem,Dependency] >= +5
 DistSumLB[Version,Dependency]: +1 dist[r,Version,Dependency] +1 dist[c,Version,Dependency] >= +7
 CenterDistSumLB: +1 center_dist[r,Rubygem] +1 center_dist[r,User] +1 center_dist[r,WebHook] +1 center_dist[r,Subscription]
  +1 center_dist[r,Ownership] +1 center_dist[r,Linkset] +1 center_dist[r,Version] +1 center_dist[r,Dependency]
  +1 center_dist[c,Rubygem] +1 center_dist[c,User] +1 center_dist[c,WebHook] +1 center_dist[c,Subscription]
  +1 center_dist[c,Ownership] +1 center_dist[c,Linkset] +1 center_dist[c,Version] +1 center_dist[c,Dependency]
  >= +22
 DistAxisLB1[r,Rubygem,User]: +1 dist[r,Rubygem,User] +7 zV[c,rt,Rubygem,User] +7 zV[c,lb,Rubygem,User]
  >= +7
 DistAxisLB1[r,User,WebHook]: +1 dist[r,User,WebHook] +7 zV[c,rt,User,WebHook] +7 zV[c,lb,User,WebHook]
  >= +7
 DistAxisLB1[r,User,Subscription]: +1 dist[r,User,Subscription] +6.5 zV[c,rt,User,Subscription] +6.5 zV[c,lb,User,Subscription]
  >= +6.5
 DistAxisLB1[r,User,Ownership]: +1 dist[r,User,Ownership] +7 zV[c,rt,User,Ownership] +7 zV[c,lb,User,Ownership]
  >= +7
 DistAxisLB1[r,Rubygem,Subscription]: +1 dist[r,Rubygem,Subscription] +4.5 zV[c,rt,Rubygem,Subscription]
  +4.5 zV[c,lb,Rubygem,Subscription] >= +4.5
 DistAxisLB1[r,Rubygem,WebHook]: +1 dist[r,Rubygem,WebHook] +5 zV[c,rt,Rubygem,WebHook] +5 zV[c,lb,Rubygem,WebHook]
  >= +5
 DistAxisLB1[r,Rubygem,Linkset]: +1 dist[r,Rubygem,Linkset] +6 zV[c,rt,Rubygem,Linkset] +6 zV[c,lb,Rubygem,Linkset]
  >= +6
 DistAxisLB1[r,Rubygem,Ownership]: +1 dist[r,Rubygem,Ownership] +5 zV[c,rt,Rubygem,Ownership] +5 zV[c,lb,Rubygem,Ownership]
  >= +5
 DistAxisLB1[r,Rubygem,Version]: +1 dist[r,Rubygem,Version] +7 zV[c,rt,Rubygem,Version] +7 zV[c,lb,Rubygem,Version]
  >= +7
 DistAxisLB1[r,Rubygem,Dependency]: +1 dist[r,Rubygem,Dependency] +5 zV[c,rt,Rubygem,Dependency] +5 zV[c,lb,Rubygem,Dependency]
  >= +5
 DistAxisLB1[r,Version,Dependency]: +1 dist[r,Version,Dependency] +7 zV[c,rt,Version,Dependency] +7 zV[c,lb,Version,Dependency]
  >= +7
 DistAxisLB1[c,Rubygem,User]: +1 dist[c,Rubygem,User] +11.5 zV[r,rt,Rubygem,User] +11.5 zV[r,lb,Rubygem,User]
  >= +11.5
 DistAxisLB1[c,User,WebHook]: +1 dist[c,User,WebHook] +12.5 zV[r,rt,User,WebHook] +12.5 zV[r,lb,User,WebHook]
  >= +12.5
 DistAxisLB1[c,User,Subscription]: +1 dist[c,User,Subscription] +11 zV[r,rt,User,Subscription] +11 zV[r,lb,User,Subscription]
  >= +11
 DistAxisLB1[c,User,Ownership]: +1 dist[c,User,Ownership] +11.5 zV[r,rt,User,Ownership] +11.5 zV[r,lb,User,Ownership]
  >= +11.5
 DistAxisLB1[c,Rubygem,Subscription]: +1 dist[c,Rubygem,Subscription] +8.5 zV[r,rt,Rubygem,Subscription]
  +8.5 zV[r,lb,Rubygem,Subscription] >= +8.5
 DistAxisLB1[c,Rubygem,WebHook]: +1 dist[c,Rubygem,WebHook] +10 zV[r,rt,Rubygem,WebHook] +10 zV[r,lb,Rubygem,WebHook]
  >= +10
 DistAxisLB1[c,Rubygem,Linkset]: +1 dist[c,Rubygem,Linkset] +8 zV[r,rt,Rubygem,Linkset] +8 zV[r,lb,Rubygem,Linkset]
  >= +8
 DistAxisLB1[c,Rubygem,Ownership]: +1 dist[c,Rubygem,Ownership] +9 zV[r,rt,Rubygem,Ownership] +9 zV[r,lb,Rubygem,Ownership]
  >= +9
 DistAxisLB1[c,Rubygem,Version]: +1 dist[c,Rubygem,Version] +10.5 zV[r,rt,Rubygem,Version] +10.5 zV[r,lb,Rubygem,Version]
  >= +10.5
 DistAxisLB1[c,Rubygem,Dependency]: +1 dist[c,Rubygem,Dependency] +9.5 zV[r,rt,Rubygem,Dependency] +9.5 zV[r,lb,Rubygem,Dependency]
  >= +9.5
 DistAxisLB1[c,Version,Dependency]: +1 dist[c,Version,Dependency] +11 zV[r,rt,Version,Dependency] +11 zV[r,lb,Version,Dependency]
  >= +11
 triangle1[Rubygem,User,WebHook,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,User,WebHook]
  <= +2
 triangle1[Rubygem,User,WebHook,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,User,WebHook]
  <= +2
 triangle1[Rubygem,User,Subscription,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,Subscription] +1 zV[r,rt,User,Subscription]
  <= +2
 triangle1[Rubygem,User,Subscription,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,Subscription] +1 zV[c,rt,User,Subscription]
  <= +2
 triangle1[Rubygem,User,Ownership,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,Ownership] +1 zV[r,rt,User,Ownership]
  <= +2
 triangle1[Rubygem,User,Ownership,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,Ownership] +1 zV[c,rt,User,Ownership]
  <= +2
 triangle1[Rubygem,User,Linkset,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,Linkset] +1 zV[r,rt,User,Linkset]
  <= +2
 triangle1[Rubygem,User,Linkset,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,Linkset] +1 zV[c,rt,User,Linkset]
  <= +2
 triangle1[Rubygem,User,Version,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,Version] +1 zV[r,rt,User,Version]
  <= +2
 triangle1[Rubygem,User,Version,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,Version] +1 zV[c,rt,User,Version]
  <= +2
 triangle1[Rubygem,User,Dependency,r]: +1 zV[r,rt,Rubygem,User] +1 zV[r,lb,Rubygem,Dependency] +1 zV[r,rt,User,Dependency]
  <= +2
 triangle1[Rubygem,User,Dependency,c]: +1 zV[c,rt,Rubygem,User] +1 zV[c,lb,Rubygem,Dependency] +1 zV[c,rt,User,Dependency]
  <= +2
 triangle1[Rubygem,WebHook,Subscription,r]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,Subscription]
  +1 zV[r,rt,WebHook,Subscription] <= +2
 triangle1[Rubygem,WebHook,Subscription,c]: +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,Rubygem,Subscription]
  +1 zV[c,rt,WebHook,Subscription] <= +2
 triangle1[Rubygem,WebHook,Ownership,r]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,Ownership] +1 zV[r,rt,WebHook,Ownership]
  <= +2
 triangle1[Rubygem,WebHook,Ownership,c]: +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,Rubygem,Ownership] +1 zV[c,rt,WebHook,Ownership]
  <= +2
 triangle1[Rubygem,WebHook,Linkset,r]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,Linkset] +1 zV[r,rt,WebHook,Linkset]
  <= +2
 triangle1[Rubygem,WebHook,Linkset,c]: +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,Rubygem,Linkset] +1 zV[c,rt,WebHook,Linkset]
  <= +2
 triangle1[Rubygem,WebHook,Version,r]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,Version] +1 zV[r,rt,WebHook,Version]
  <= +2
 triangle1[Rubygem,WebHook,Version,c]: +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,Rubygem,Version] +1 zV[c,rt,WebHook,Version]
  <= +2
 triangle1[Rubygem,WebHook,Dependency,r]: +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,Rubygem,Dependency] +1 zV[r,rt,WebHook,Dependency]
  <= +2
 triangle1[Rubygem,WebHook,Dependency,c]: +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,Rubygem,Dependency] +1 zV[c,rt,WebHook,Dependency]
  <= +2
 triangle1[Rubygem,Subscription,Ownership,r]: +1 zV[r,rt,Rubygem,Subscription] +1 zV[r,lb,Rubygem,Ownership]
  +1 zV[r,rt,Subscription,Ownership] <= +2
 triangle1[Rubygem,Subscription,Ownership,c]: +1 zV[c,rt,Rubygem,Subscription] +1 zV[c,lb,Rubygem,Ownership]
  +1 zV[c,rt,Subscription,Ownership] <= +2
 triangle1[Rubygem,Subscription,Linkset,r]: +1 zV[r,rt,Rubygem,Subscription] +1 zV[r,lb,Rubygem,Linkset]
  +1 zV[r,rt,Subscription,Linkset] <= +2
 triangle1[Rubygem,Subscription,Linkset,c]: +1 zV[c,rt,Rubygem,Subscription] +1 zV[c,lb,Rubygem,Linkset]
  +1 zV[c,rt,Subscription,Linkset] <= +2
 triangle1[Rubygem,Subscription,Version,r]: +1 zV[r,rt,Rubygem,Subscription] +1 zV[r,lb,Rubygem,Version]
  +1 zV[r,rt,Subscription,Version] <= +2
 triangle1[Rubygem,Subscription,Version,c]: +1 zV[c,rt,Rubygem,Subscription] +1 zV[c,lb,Rubygem,Version]
  +1 zV[c,rt,Subscription,Version] <= +2
 triangle1[Rubygem,Subscription,Dependency,r]: +1 zV[r,rt,Rubygem,Subscription] +1 zV[r,lb,Rubygem,Dependency]
  +1 zV[r,rt,Subscription,Dependency] <= +2
 triangle1[Rubygem,Subscription,Dependency,c]: +1 zV[c,rt,Rubygem,Subscription] +1 zV[c,lb,Rubygem,Dependency]
  +1 zV[c,rt,Subscription,Dependency] <= +2
 triangle1[Rubygem,Ownership,Linkset,r]: +1 zV[r,rt,Rubygem,Ownership] +1 zV[r,lb,Rubygem,Linkset] +1 zV[r,rt,Ownership,Linkset]
  <= +2
 triangle1[Rubygem,Ownership,Linkset,c]: +1 zV[c,rt,Rubygem,Ownership] +1 zV[c,lb,Rubygem,Linkset] +1 zV[c,rt,Ownership,Linkset]
  <= +2
 triangle1[Rubygem,Ownership,Version,r]: +1 zV[r,rt,Rubygem,Ownership] +1 zV[r,lb,Rubygem,Version] +1 zV[r,rt,Ownership,Version]
  <= +2
 triangle1[Rubygem,Ownership,Version,c]: +1 zV[c,rt,Rubygem,Ownership] +1 zV[c,lb,Rubygem,Version] +1 zV[c,rt,Ownership,Version]
  <= +2
 triangle1[Rubygem,Ownership,Dependency,r]: +1 zV[r,rt,Rubygem,Ownership] +1 zV[r,lb,Rubygem,Dependency]
  +1 zV[r,rt,Ownership,Dependency] <= +2
 triangle1[Rubygem,Ownership,Dependency,c]: +1 zV[c,rt,Rubygem,Ownership] +1 zV[c,lb,Rubygem,Dependency]
  +1 zV[c,rt,Ownership,Dependency] <= +2
 triangle1[Rubygem,Linkset,Version,r]: +1 zV[r,rt,Rubygem,Linkset] +1 zV[r,lb,Rubygem,Version] +1 zV[r,rt,Linkset,Version]
  <= +2
 triangle1[Rubygem,Linkset,Version,c]: +1 zV[c,rt,Rubygem,Linkset] +1 zV[c,lb,Rubygem,Version] +1 zV[c,rt,Linkset,Version]
  <= +2
 triangle1[Rubygem,Linkset,Dependency,r]: +1 zV[r,rt,Rubygem,Linkset] +1 zV[r,lb,Rubygem,Dependency] +1 zV[r,rt,Linkset,Dependency]
  <= +2
 triangle1[Rubygem,Linkset,Dependency,c]: +1 zV[c,rt,Rubygem,Linkset] +1 zV[c,lb,Rubygem,Dependency] +1 zV[c,rt,Linkset,Dependency]
  <= +2
 triangle1[Rubygem,Version,Dependency,r]: +1 zV[r,rt,Rubygem,Version] +1 zV[r,lb,Rubygem,Dependency] +1 zV[r,rt,Version,Dependency]
  <= +2
 triangle1[Rubygem,Version,Dependency,c]: +1 zV[c,rt,Rubygem,Version] +1 zV[c,lb,Rubygem,Dependency] +1 zV[c,rt,Version,Dependency]
  <= +2
 triangle1[User,WebHook,Subscription,r]: +1 zV[r,rt,User,WebHook] +1 zV[r,lb,User,Subscription] +1 zV[r,rt,WebHook,Subscription]
  <= +2
 triangle1[User,WebHook,Subscription,c]: +1 zV[c,rt,User,WebHook] +1 zV[c,lb,User,Subscription] +1 zV[c,rt,WebHook,Subscription]
  <= +2
 triangle1[User,WebHook,Ownership,r]: +1 zV[r,rt,User,WebHook] +1 zV[r,lb,User,Ownership] +1 zV[r,rt,WebHook,Ownership]
  <= +2
 triangle1[User,WebHook,Ownership,c]: +1 zV[c,rt,User,WebHook] +1 zV[c,lb,User,Ownership] +1 zV[c,rt,WebHook,Ownership]
  <= +2
 triangle1[User,WebHook,Linkset,r]: +1 zV[r,rt,User,WebHook] +1 zV[r,lb,User,Linkset] +1 zV[r,rt,WebHook,Linkset]
  <= +2
 triangle1[User,WebHook,Linkset,c]: +1 zV[c,rt,User,WebHook] +1 zV[c,lb,User,Linkset] +1 zV[c,rt,WebHook,Linkset]
  <= +2
 triangle1[User,WebHook,Version,r]: +1 zV[r,rt,User,WebHook] +1 zV[r,lb,User,Version] +1 zV[r,rt,WebHook,Version]
  <= +2
 triangle1[User,WebHook,Version,c]: +1 zV[c,rt,User,WebHook] +1 zV[c,lb,User,Version] +1 zV[c,rt,WebHook,Version]
  <= +2
 triangle1[User,WebHook,Dependency,r]: +1 zV[r,rt,User,WebHook] +1 zV[r,lb,User,Dependency] +1 zV[r,rt,WebHook,Dependency]
  <= +2
 triangle1[User,WebHook,Dependency,c]: +1 zV[c,rt,User,WebHook] +1 zV[c,lb,User,Dependency] +1 zV[c,rt,WebHook,Dependency]
  <= +2
 triangle1[User,Subscription,Ownership,r]: +1 zV[r,rt,User,Subscription] +1 zV[r,lb,User,Ownership] +1 zV[r,rt,Subscription,Ownership]
  <= +2
 triangle1[User,Subscription,Ownership,c]: +1 zV[c,rt,User,Subscription] +1 zV[c,lb,User,Ownership] +1 zV[c,rt,Subscription,Ownership]
  <= +2
 triangle1[User,Subscription,Linkset,r]: +1 zV[r,rt,User,Subscription] +1 zV[r,lb,User,Linkset] +1 zV[r,rt,Subscription,Linkset]
  <= +2
 triangle1[User,Subscription,Linkset,c]: +1 zV[c,rt,User,Subscription] +1 zV[c,lb,User,Linkset] +1 zV[c,rt,Subscription,Linkset]
  <= +2
 triangle1[User,Subscription,Version,r]: +1 zV[r,rt,User,Subscription] +1 zV[r,lb,User,Version] +1 zV[r,rt,Subscription,Version]
  <= +2
 triangle1[User,Subscription,Version,c]: +1 zV[c,rt,User,Subscription] +1 zV[c,lb,User,Version] +1 zV[c,rt,Subscription,Version]
  <= +2
 triangle1[User,Subscription,Dependency,r]: +1 zV[r,rt,User,Subscription] +1 zV[r,lb,User,Dependency]
  +1 zV[r,rt,Subscription,Dependency] <= +2
 triangle1[User,Subscription,Dependency,c]: +1 zV[c,rt,User,Subscription] +1 zV[c,lb,User,Dependency]
  +1 zV[c,rt,Subscription,Dependency] <= +2
 triangle1[User,Ownership,Linkset,r]: +1 zV[r,rt,User,Ownership] +1 zV[r,lb,User,Linkset] +1 zV[r,rt,Ownership,Linkset]
  <= +2
 triangle1[User,Ownership,Linkset,c]: +1 zV[c,rt,User,Ownership] +1 zV[c,lb,User,Linkset] +1 zV[c,rt,Ownership,Linkset]
  <= +2
 triangle1[User,Ownership,Version,r]: +1 zV[r,rt,User,Ownership] +1 zV[r,lb,User,Version] +1 zV[r,rt,Ownership,Version]
  <= +2
 triangle1[User,Ownership,Version,c]: +1 zV[c,rt,User,Ownership] +1 zV[c,lb,User,Version] +1 zV[c,rt,Ownership,Version]
  <= +2
 triangle1[User,Ownership,Dependency,r]: +1 zV[r,rt,User,Ownership] +1 zV[r,lb,User,Dependency] +1 zV[r,rt,Ownership,Dependency]
  <= +2
 triangle1[User,Ownership,Dependency,c]: +1 zV[c,rt,User,Ownership] +1 zV[c,lb,User,Dependency] +1 zV[c,rt,Ownership,Dependency]
  <= +2
 triangle1[User,Linkset,Version,r]: +1 zV[r,rt,User,Linkset] +1 zV[r,lb,User,Version] +1 zV[r,rt,Linkset,Version]
  <= +2
 triangle1[User,Linkset,Version,c]: +1 zV[c,rt,User,Linkset] +1 zV[c,lb,User,Version] +1 zV[c,rt,Linkset,Version]
  <= +2
 triangle1[User,Linkset,Dependency,r]: +1 zV[r,rt,User,Linkset] +1 zV[r,lb,User,Dependency] +1 zV[r,rt,Linkset,Dependency]
  <= +2
 triangle1[User,Linkset,Dependency,c]: +1 zV[c,rt,User,Linkset] +1 zV[c,lb,User,Dependency] +1 zV[c,rt,Linkset,Dependency]
  <= +2
 triangle1[User,Version,Dependency,r]: +1 zV[r,rt,User,Version] +1 zV[r,lb,User,Dependency] +1 zV[r,rt,Version,Dependency]
  <= +2
 triangle1[User,Version,Dependency,c]: +1 zV[c,rt,User,Version] +1 zV[c,lb,User,Dependency] +1 zV[c,rt,Version,Dependency]
  <= +2
 triangle1[WebHook,Subscription,Ownership,r]: +1 zV[r,rt,WebHook,Subscription] +1 zV[r,lb,WebHook,Ownership]
  +1 zV[r,rt,Subscription,Ownership] <= +2
 triangle1[WebHook,Subscription,Ownership,c]: +1 zV[c,rt,WebHook,Subscription] +1 zV[c,lb,WebHook,Ownership]
  +1 zV[c,rt,Subscription,Ownership] <= +2
 triangle1[WebHook,Subscription,Linkset,r]: +1 zV[r,rt,WebHook,Subscription] +1 zV[r,lb,WebHook,Linkset]
  +1 zV[r,rt,Subscription,Linkset] <= +2
 triangle1[WebHook,Subscription,Linkset,c]: +1 zV[c,rt,WebHook,Subscription] +1 zV[c,lb,WebHook,Linkset]
  +1 zV[c,rt,Subscription,Linkset] <= +2
 triangle1[WebHook,Subscription,Version,r]: +1 zV[r,rt,WebHook,Subscription] +1 zV[r,lb,WebHook,Version]
  +1 zV[r,rt,Subscription,Version] <= +2
 triangle1[WebHook,Subscription,Version,c]: +1 zV[c,rt,WebHook,Subscription] +1 zV[c,lb,WebHook,Version]
  +1 zV[c,rt,Subscription,Version] <= +2
 triangle1[WebHook,Subscription,Dependency,r]: +1 zV[r,rt,WebHook,Subscription] +1 zV[r,lb,WebHook,Dependency]
  +1 zV[r,rt,Subscription,Dependency] <= +2
 triangle1[WebHook,Subscription,Dependency,c]: +1 zV[c,rt,WebHook,Subscription] +1 zV[c,lb,WebHook,Dependency]
  +1 zV[c,rt,Subscription,Dependency] <= +2
 triangle1[WebHook,Ownership,Linkset,r]: +1 zV[r,rt,WebHook,Ownership] +1 zV[r,lb,WebHook,Linkset] +1 zV[r,rt,Ownership,Linkset]
  <= +2
 triangle1[WebHook,Ownership,Linkset,c]: +1 zV[c,rt,WebHook,Ownership] +1 zV[c,lb,WebHook,Linkset] +1 zV[c,rt,Ownership,Linkset]
  <= +2
 triangle1[WebHook,Ownership,Version,r]: +1 zV[r,rt,WebHook,Ownership] +1 zV[r,lb,WebHook,Version] +1 zV[r,rt,Ownership,Version]
  <= +2
 triangle1[WebHook,Ownership,Version,c]: +1 zV[c,rt,WebHook,Ownership] +1 zV[c,lb,WebHook,Version] +1 zV[c,rt,Ownership,Version]
  <= +2
 triangle1[WebHook,Ownership,Dependency,r]: +1 zV[r,rt,WebHook,Ownership] +1 zV[r,lb,WebHook,Dependency]
  +1 zV[r,rt,Ownership,Dependency] <= +2
 triangle1[WebHook,Ownership,Dependency,c]: +1 zV[c,rt,WebHook,Ownership] +1 zV[c,lb,WebHook,Dependency]
  +1 zV[c,rt,Ownership,Dependency] <= +2
 triangle1[WebHook,Linkset,Version,r]: +1 zV[r,rt,WebHook,Linkset] +1 zV[r,lb,WebHook,Version] +1 zV[r,rt,Linkset,Version]
  <= +2
 triangle1[WebHook,Linkset,Version,c]: +1 zV[c,rt,WebHook,Linkset] +1 zV[c,lb,WebHook,Version] +1 zV[c,rt,Linkset,Version]
  <= +2
 triangle1[WebHook,Linkset,Dependency,r]: +1 zV[r,rt,WebHook,Linkset] +1 zV[r,lb,WebHook,Dependency] +1 zV[r,rt,Linkset,Dependency]
  <= +2
 triangle1[WebHook,Linkset,Dependency,c]: +1 zV[c,rt,WebHook,Linkset] +1 zV[c,lb,WebHook,Dependency] +1 zV[c,rt,Linkset,Dependency]
  <= +2
 triangle1[WebHook,Version,Dependency,r]: +1 zV[r,rt,WebHook,Version] +1 zV[r,lb,WebHook,Dependency] +1 zV[r,rt,Version,Dependency]
  <= +2
 triangle1[WebHook,Version,Dependency,c]: +1 zV[c,rt,WebHook,Version] +1 zV[c,lb,WebHook,Dependency] +1 zV[c,rt,Version,Dependency]
  <= +2
 triangle1[Subscription,Ownership,Linkset,r]: +1 zV[r,rt,Subscription,Ownership] +1 zV[r,lb,Subscription,Linkset]
  +1 zV[r,rt,Ownership,Linkset] <= +2
 triangle1[Subscription,Ownership,Linkset,c]: +1 zV[c,rt,Subscription,Ownership] +1 zV[c,lb,Subscription,Linkset]
  +1 zV[c,rt,Ownership,Linkset] <= +2
 triangle1[Subscription,Ownership,Version,r]: +1 zV[r,rt,Subscription,Ownership] +1 zV[r,lb,Subscription,Version]
  +1 zV[r,rt,Ownership,Version] <= +2
 triangle1[Subscription,Ownership,Version,c]: +1 zV[c,rt,Subscription,Ownership] +1 zV[c,lb,Subscription,Version]
  +1 zV[c,rt,Ownership,Version] <= +2
 triangle1[Subscription,Ownership,Dependency,r]: +1 zV[r,rt,Subscription,Ownership] +1 zV[r,lb,Subscription,Dependency]
  +1 zV[r,rt,Ownership,Dependency] <= +2
 triangle1[Subscription,Ownership,Dependency,c]: +1 zV[c,rt,Subscription,Ownership] +1 zV[c,lb,Subscription,Dependency]
  +1 zV[c,rt,Ownership,Dependency] <= +2
 triangle1[Subscription,Linkset,Version,r]: +1 zV[r,rt,Subscription,Linkset] +1 zV[r,lb,Subscription,Version]
  +1 zV[r,rt,Linkset,Version] <= +2
 triangle1[Subscription,Linkset,Version,c]: +1 zV[c,rt,Subscription,Linkset] +1 zV[c,lb,Subscription,Version]
  +1 zV[c,rt,Linkset,Version] <= +2
 triangle1[Subscription,Linkset,Dependency,r]: +1 zV[r,rt,Subscription,Linkset] +1 zV[r,lb,Subscription,Dependency]
  +1 zV[r,rt,Linkset,Dependency] <= +2
 triangle1[Subscription,Linkset,Dependency,c]: +1 zV[c,rt,Subscription,Linkset] +1 zV[c,lb,Subscription,Dependency]
  +1 zV[c,rt,Linkset,Dependency] <= +2
 triangle1[Subscription,Version,Dependency,r]: +1 zV[r,rt,Subscription,Version] +1 zV[r,lb,Subscription,Dependency]
  +1 zV[r,rt,Version,Dependency] <= +2
 triangle1[Subscription,Version,Dependency,c]: +1 zV[c,rt,Subscription,Version] +1 zV[c,lb,Subscription,Dependency]
  +1 zV[c,rt,Version,Dependency] <= +2
 triangle1[Ownership,Linkset,Version,r]: +1 zV[r,rt,Ownership,Linkset] +1 zV[r,lb,Ownership,Version] +1 zV[r,rt,Linkset,Version]
  <= +2
 triangle1[Ownership,Linkset,Version,c]: +1 zV[c,rt,Ownership,Linkset] +1 zV[c,lb,Ownership,Version] +1 zV[c,rt,Linkset,Version]
  <= +2
 triangle1[Ownership,Linkset,Dependency,r]: +1 zV[r,rt,Ownership,Linkset] +1 zV[r,lb,Ownership,Dependency]
  +1 zV[r,rt,Linkset,Dependency] <= +2
 triangle1[Ownership,Linkset,Dependency,c]: +1 zV[c,rt,Ownership,Linkset] +1 zV[c,lb,Ownership,Dependency]
  +1 zV[c,rt,Linkset,Dependency] <= +2
 triangle1[Ownership,Version,Dependency,r]: +1 zV[r,rt,Ownership,Version] +1 zV[r,lb,Ownership,Dependency]
  +1 zV[r,rt,Version,Dependency] <= +2
 triangle1[Ownership,Version,Dependency,c]: +1 zV[c,rt,Ownership,Version] +1 zV[c,lb,Ownership,Dependency]
  +1 zV[c,rt,Version,Dependency] <= +2
 triangle1[Linkset,Version,Dependency,r]: +1 zV[r,rt,Linkset,Version] +1 zV[r,lb,Linkset,Dependency] +1 zV[r,rt,Version,Dependency]
  <= +2
 triangle1[Linkset,Version,Dependency,c]: +1 zV[c,rt,Linkset,Version] +1 zV[c,lb,Linkset,Dependency] +1 zV[c,rt,Version,Dependency]
  <= +2
 triangle2[Rubygem,User,WebHook,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,WebHook] +1 zV[r,lb,User,WebHook]
  <= +2
 triangle2[Rubygem,User,WebHook,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,WebHook] +1 zV[c,lb,User,WebHook]
  <= +2
 triangle2[Rubygem,User,Subscription,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,Subscription] +1 zV[r,lb,User,Subscription]
  <= +2
 triangle2[Rubygem,User,Subscription,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,Subscription] +1 zV[c,lb,User,Subscription]
  <= +2
 triangle2[Rubygem,User,Ownership,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,Ownership] +1 zV[r,lb,User,Ownership]
  <= +2
 triangle2[Rubygem,User,Ownership,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,Ownership] +1 zV[c,lb,User,Ownership]
  <= +2
 triangle2[Rubygem,User,Linkset,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,Linkset] +1 zV[r,lb,User,Linkset]
  <= +2
 triangle2[Rubygem,User,Linkset,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,Linkset] +1 zV[c,lb,User,Linkset]
  <= +2
 triangle2[Rubygem,User,Version,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,Version] +1 zV[r,lb,User,Version]
  <= +2
 triangle2[Rubygem,User,Version,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,Version] +1 zV[c,lb,User,Version]
  <= +2
 triangle2[Rubygem,User,Dependency,r]: +1 zV[r,lb,Rubygem,User] +1 zV[r,rt,Rubygem,Dependency] +1 zV[r,lb,User,Dependency]
  <= +2
 triangle2[Rubygem,User,Dependency,c]: +1 zV[c,lb,Rubygem,User] +1 zV[c,rt,Rubygem,Dependency] +1 zV[c,lb,User,Dependency]
  <= +2
 triangle2[Rubygem,WebHook,Subscription,r]: +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,Rubygem,Subscription]
  +1 zV[r,lb,WebHook,Subscription] <= +2
 triangle2[Rubygem,WebHook,Subscription,c]: +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,Rubygem,Subscription]
  +1 zV[c,lb,WebHook,Subscription] <= +2
 triangle2[Rubygem,WebHook,Ownership,r]: +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,Rubygem,Ownership] +1 zV[r,lb,WebHook,Ownership]
  <= +2
 triangle2[Rubygem,WebHook,Ownership,c]: +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,Rubygem,Ownership] +1 zV[c,lb,WebHook,Ownership]
  <= +2
 triangle2[Rubygem,WebHook,Linkset,r]: +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,Rubygem,Linkset] +1 zV[r,lb,WebHook,Linkset]
  <= +2
 triangle2[Rubygem,WebHook,Linkset,c]: +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,Rubygem,Linkset] +1 zV[c,lb,WebHook,Linkset]
  <= +2
 triangle2[Rubygem,WebHook,Version,r]: +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,Rubygem,Version] +1 zV[r,lb,WebHook,Version]
  <= +2
 triangle2[Rubygem,WebHook,Version,c]: +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,Rubygem,Version] +1 zV[c,lb,WebHook,Version]
  <= +2
 triangle2[Rubygem,WebHook,Dependency,r]: +1 zV[r,lb,Rubygem,WebHook] +1 zV[r,rt,Rubygem,Dependency] +1 zV[r,lb,WebHook,Dependency]
  <= +2
 triangle2[Rubygem,WebHook,Dependency,c]: +1 zV[c,lb,Rubygem,WebHook] +1 zV[c,rt,Rubygem,Dependency] +1 zV[c,lb,WebHook,Dependency]
  <= +2
 triangle2[Rubygem,Subscription,Ownership,r]: +1 zV[r,lb,Rubygem,Subscription] +1 zV[r,rt,Rubygem,Ownership]
  +1 zV[r,lb,Subscription,Ownership] <= +2
 triangle2[Rubygem,Subscription,Ownership,c]: +1 zV[c,lb,Rubygem,Subscription] +1 zV[c,rt,Rubygem,Ownership]
  +1 zV[c,lb,Subscription,Ownership] <= +2
 triangle2[Rubygem,Subscription,Linkset,r]: +1 zV[r,lb,Rubygem,Subscription] +1 zV[r,rt,Rubygem,Linkset]
  +1 zV[r,lb,Subscription,Linkset] <= +2
 triangle2[Rubygem,Subscription,Linkset,c]: +1 zV[c,lb,Rubygem,Subscription] +1 zV[c,rt,Rubygem,Linkset]
  +1 zV[c,lb,Subscription,Linkset] <= +2
 triangle2[Rubygem,Subscription,Version,r]: +1 zV[r,lb,Rubygem,Subscription] +1 zV[r,rt,Rubygem,Version]
  +1 zV[r,lb,Subscription,Version] <= +2
 triangle2[Rubygem,Subscription,Version,c]: +1 zV[c,lb,Rubygem,Subscription] +1 zV[c,rt,Rubygem,Version]
  +1 zV[c,lb,Subscription,Version] <= +2
 triangle2[Rubygem,Subscription,Dependency,r]: +1 zV[r,lb,Rubygem,Subscription] +1 zV[r,rt,Rubygem,Dependency]
  +1 zV[r,lb,Subscription,Dependency] <= +2
 triangle2[Rubygem,Subscription,Dependency,c]: +1 zV[c,lb,Rubygem,Subscription] +1 zV[c,rt,Rubygem,Dependency]
  +1 zV[c,lb,Subscription,Dependency] <= +2
 triangle2[Rubygem,Ownership,Linkset,r]: +1 zV[r,lb,Rubygem,Ownership] +1 zV[r,rt,Rubygem,Linkset] +1 zV[r,lb,Ownership,Linkset]
  <= +2
 triangle2[Rubygem,Ownership,Linkset,c]: +1 zV[c,lb,Rubygem,Ownership] +1 zV[c,rt,Rubygem,Linkset] +1 zV[c,lb,Ownership,Linkset]
  <= +2
 triangle2[Rubygem,Ownership,Version,r]: +1 zV[r,lb,Rubygem,Ownership] +1 zV[r,rt,Rubygem,Version] +1 zV[r,lb,Ownership,Version]
  <= +2
 triangle2[Rubygem,Ownership,Version,c]: +1 zV[c,lb,Rubygem,Ownership] +1 zV[c,rt,Rubygem,Version] +1 zV[c,lb,Ownership,Version]
  <= +2
 triangle2[Rubygem,Ownership,Dependency,r]: +1 zV[r,lb,Rubygem,Ownership] +1 zV[r,rt,Rubygem,Dependency]
  +1 zV[r,lb,Ownership,Dependency] <= +2
 triangle2[Rubygem,Ownership,Dependency,c]: +1 zV[c,lb,Rubygem,Ownership] +1 zV[c,rt,Rubygem,Dependency]
  +1 zV[c,lb,Ownership,Dependency] <= +2
 triangle2[Rubygem,Linkset,Version,r]: +1 zV[r,lb,Rubygem,Linkset] +1 zV[r,rt,Rubygem,Version] +1 zV[r,lb,Linkset,Version]
  <= +2
 triangle2[Rubygem,Linkset,Version,c]: +1 zV[c,lb,Rubygem,Linkset] +1 zV[c,rt,Rubygem,Version] +1 zV[c,lb,Linkset,Version]
  <= +2
 triangle2[Rubygem,Linkset,Dependency,r]: +1 zV[r,lb,Rubygem,Linkset] +1 zV[r,rt,Rubygem,Dependency] +1 zV[r,lb,Linkset,Dependency]
  <= +2
 triangle2[Rubygem,Linkset,Dependency,c]: +1 zV[c,lb,Rubygem,Linkset] +1 zV[c,rt,Rubygem,Dependency] +1 zV[c,lb,Linkset,Dependency]
  <= +2
 triangle2[Rubygem,Version,Dependency,r]: +1 zV[r,lb,Rubygem,Version] +1 zV[r,rt,Rubygem,Dependency] +1 zV[r,lb,Version,Dependency]
  <= +2
 triangle2[Rubygem,Version,Dependency,c]: +1 zV[c,lb,Rubygem,Version] +1 zV[c,rt,Rubygem,Dependency] +1 zV[c,lb,Version,Dependency]
  <= +2
 triangle2[User,WebHook,Subscription,r]: +1 zV[r,lb,User,WebHook] +1 zV[r,rt,User,Subscription] +1 zV[r,lb,WebHook,Subscription]
  <= +2
 triangle2[User,WebHook,Subscription,c]: +1 zV[c,lb,User,WebHook] +1 zV[c,rt,User,Subscription] +1 zV[c,lb,WebHook,Subscription]
  <= +2
 triangle2[User,WebHook,Ownership,r]: +1 zV[r,lb,User,WebHook] +1 zV[r,rt,User,Ownership] +1 zV[r,lb,WebHook,Ownership]
  <= +2
 triangle2[User,WebHook,Ownership,c]: +1 zV[c,lb,User,WebHook] +1 zV[c,rt,User,Ownership] +1 zV[c,lb,WebHook,Ownership]
  <= +2
 triangle2[User,WebHook,Linkset,r]: +1 zV[r,lb,User,WebHook] +1 zV[r,rt,User,Linkset] +1 zV[r,lb,WebHook,Linkset]
  <= +2
 triangle2[User,WebHook,Linkset,c]: +1 zV[c,lb,User,WebHook] +1 zV[c,rt,User,Linkset] +1 zV[c,lb,WebHook,Linkset]
  <= +2
 triangle2[User,WebHook,Version,r]: +1 zV[r,lb,User,WebHook] +1 zV[r,rt,User,Version] +1 zV[r,lb,WebHook,Version]
  <= +2
 triangle2[User,WebHook,Version,c]: +1 zV[c,lb,User,WebHook] +1 zV[c,rt,User,Version] +1 zV[c,lb,WebHook,Version]
  <= +2
 triangle2[User,WebHook,Dependency,r]: +1 zV[r,lb,User,WebHook] +1 zV[r,rt,User,Dependency] +1 zV[r,lb,WebHook,Dependency]
  <= +2
 triangle2[User,WebHook,Dependency,c]: +1 zV[c,lb,User,WebHook] +1 zV[c,rt,User,Dependency] +1 zV[c,lb,WebHook,Dependency]
  <= +2
 triangle2[User,Subscription,Ownership,r]: +1 zV[r,lb,User,Subscription] +1 zV[r,rt,User,Ownership] +1 zV[r,lb,Subscription,Ownership]
  <= +2
 triangle2[User,Subscription,Ownership,c]: +1 zV[c,lb,User,Subscription] +1 zV[c,rt,User,Ownership] +1 zV[c,lb,Subscription,Ownership]
  <= +2
 triangle2[User,Subscription,Linkset,r]: +1 zV[r,lb,User,Subscription] +1 zV[r,rt,User,Linkset] +1 zV[r,lb,Subscription,Linkset]
  <= +2
 triangle2[User,Subscription,Linkset,c]: +1 zV[c,lb,User,Subscription] +1 zV[c,rt,User,Linkset] +1 zV[c,lb,Subscription,Linkset]
  <= +2
 triangle2[User,Subscription,Version,r]: +1 zV[r,lb,User,Subscription] +1 zV[r,rt,User,Version] +1 zV[r,lb,Subscription,Version]
  <= +2
 triangle2[User,Subscription,Version,c]: +1 zV[c,lb,User,Subscription] +1 zV[c,rt,User,Version] +1 zV[c,lb,Subscription,Version]
  <= +2
 triangle2[User,Subscription,Dependency,r]: +1 zV[r,lb,User,Subscription] +1 zV[r,rt,User,Dependency]
  +1 zV[r,lb,Subscription,Dependency] <= +2
 triangle2[User,Subscription,Dependency,c]: +1 zV[c,lb,User,Subscription] +1 zV[c,rt,User,Dependency]
  +1 zV[c,lb,Subscription,Dependency] <= +2
 triangle2[User,Ownership,Linkset,r]: +1 zV[r,lb,User,Ownership] +1 zV[r,rt,User,Linkset] +1 zV[r,lb,Ownership,Linkset]
  <= +2
 triangle2[User,Ownership,Linkset,c]: +1 zV[c,lb,User,Ownership] +1 zV[c,rt,User,Linkset] +1 zV[c,lb,Ownership,Linkset]
  <= +2
 triangle2[User,Ownership,Version,r]: +1 zV[r,lb,User,Ownership] +1 zV[r,rt,User,Version] +1 zV[r,lb,Ownership,Version]
  <= +2
 triangle2[User,Ownership,Version,c]: +1 zV[c,lb,User,Ownership] +1 zV[c,rt,User,Version] +1 zV[c,lb,Ownership,Version]
  <= +2
 triangle2[User,Ownership,Dependency,r]: +1 zV[r,lb,User,Ownership] +1 zV[r,rt,User,Dependency] +1 zV[r,lb,Ownership,Dependency]
  <= +2
 triangle2[User,Ownership,Dependency,c]: +1 zV[c,lb,User,Ownership] +1 zV[c,rt,User,Dependency] +1 zV[c,lb,Ownership,Dependency]
  <= +2
 triangle2[User,Linkset,Version,r]: +1 zV[r,lb,User,Linkset] +1 zV[r,rt,User,Version] +1 zV[r,lb,Linkset,Version]
  <= +2
 triangle2[User,Linkset,Version,c]: +1 zV[c,lb,User,Linkset] +1 zV[c,rt,User,Version] +1 zV[c,lb,Linkset,Version]
  <= +2
 triangle2[User,Linkset,Dependency,r]: +1 zV[r,lb,User,Linkset] +1 zV[r,rt,User,Dependency] +1 zV[r,lb,Linkset,Dependency]
  <= +2
 triangle2[User,Linkset,Dependency,c]: +1 zV[c,lb,User,Linkset] +1 zV[c,rt,User,Dependency] +1 zV[c,lb,Linkset,Dependency]
  <= +2
 triangle2[User,Version,Dependency,r]: +1 zV[r,lb,User,Version] +1 zV[r,rt,User,Dependency] +1 zV[r,lb,Version,Dependency]
  <= +2
 triangle2[User,Version,Dependency,c]: +1 zV[c,lb,User,Version] +1 zV[c,rt,User,Dependency] +1 zV[c,lb,Version,Dependency]
  <= +2
 triangle2[WebHook,Subscription,Ownership,r]: +1 zV[r,lb,WebHook,Subscription] +1 zV[r,rt,WebHook,Ownership]
  +1 zV[r,lb,Subscription,Ownership] <= +2
 triangle2[WebHook,Subscription,Ownership,c]: +1 zV[c,lb,WebHook,Subscription] +1 zV[c,rt,WebHook,Ownership]
  +1 zV[c,lb,Subscription,Ownership] <= +2
 triangle2[WebHook,Subscription,Linkset,r]: +1 zV[r,lb,WebHook,Subscription] +1 zV[r,rt,WebHook,Linkset]
  +1 zV[r,lb,Subscription,Linkset] <= +2
 triangle2[WebHook,Subscription,Linkset,c]: +1 zV[c,lb,WebHook,Subscription] +1 zV[c,rt,WebHook,Linkset]
  +1 zV[c,lb,Subscription,Linkset] <= +2
 triangle2[WebHook,Subscription,Version,r]: +1 zV[r,lb,WebHook,Subscription] +1 zV[r,rt,WebHook,Version]
  +1 zV[r,lb,Subscription,Version] <= +2
 triangle2[WebHook,Subscription,Version,c]: +1 zV[c,lb,WebHook,Subscription] +1 zV[c,rt,WebHook,Version]
  +1 zV[c,lb,Subscription,Version] <= +2
 triangle2[WebHook,Subscription,Dependency,r]: +1 zV[r,lb,WebHook,Subscription] +1 zV[r,rt,WebHook,Dependency]
  +1 zV[r,lb,Subscription,Dependency] <= +2
 triangle2[WebHook,Subscription,Dependency,c]: +1 zV[c,lb,WebHook,Subscription] +1 zV[c,rt,WebHook,Dependency]
  +1 zV[c,lb,Subscription,Dependency] <= +2
 triangle2[WebHook,Ownership,Linkset,r]: +1 zV[r,lb,WebHook,Ownership] +1 zV[r,rt,WebHook,Linkset] +1 zV[r,lb,Ownership,Linkset]
  <= +2
 triangle2[WebHook,Ownership,Linkset,c]: +1 zV[c,lb,WebHook,Ownership] +1 zV[c,rt,WebHook,Linkset] +1 zV[c,lb,Ownership,Linkset]
  <= +2
 triangle2[WebHook,Ownership,Version,r]: +1 zV[r,lb,WebHook,Ownership] +1 zV[r,rt,WebHook,Version] +1 zV[r,lb,Ownership,Version]
  <= +2
 triangle2[WebHook,Ownership,Version,c]: +1 zV[c,lb,WebHook,Ownership] +1 zV[c,rt,WebHook,Version] +1 zV[c,lb,Ownership,Version]
  <= +2
 triangle2[WebHook,Ownership,Dependency,r]: +1 zV[r,lb,WebHook,Ownership] +1 zV[r,rt,WebHook,Dependency]
  +1 zV[r,lb,Ownership,Dependency] <= +2
 triangle2[WebHook,Ownership,Dependency,c]: +1 zV[c,lb,WebHook,Ownership] +1 zV[c,rt,WebHook,Dependency]
  +1 zV[c,lb,Ownership,Dependency] <= +2
 triangle2[WebHook,Linkset,Version,r]: +1 zV[r,lb,WebHook,Linkset] +1 zV[r,rt,WebHook,Version] +1 zV[r,lb,Linkset,Version]
  <= +2
 triangle2[WebHook,Linkset,Version,c]: +1 zV[c,lb,WebHook,Linkset] +1 zV[c,rt,WebHook,Version] +1 zV[c,lb,Linkset,Version]
  <= +2
 triangle2[WebHook,Linkset,Dependency,r]: +1 zV[r,lb,WebHook,Linkset] +1 zV[r,rt,WebHook,Dependency] +1 zV[r,lb,Linkset,Dependency]
  <= +2
 triangle2[WebHook,Linkset,Dependency,c]: +1 zV[c,lb,WebHook,Linkset] +1 zV[c,rt,WebHook,Dependency] +1 zV[c,lb,Linkset,Dependency]
  <= +2
 triangle2[WebHook,Version,Dependency,r]: +1 zV[r,lb,WebHook,Version] +1 zV[r,rt,WebHook,Dependency] +1 zV[r,lb,Version,Dependency]
  <= +2
 triangle2[WebHook,Version,Dependency,c]: +1 zV[c,lb,WebHook,Version] +1 zV[c,rt,WebHook,Dependency] +1 zV[c,lb,Version,Dependency]
  <= +2
 triangle2[Subscription,Ownership,Linkset,r]: +1 zV[r,lb,Subscription,Ownership] +1 zV[r,rt,Subscription,Linkset]
  +1 zV[r,lb,Ownership,Linkset] <= +2
 triangle2[Subscription,Ownership,Linkset,c]: +1 zV[c,lb,Subscription,Ownership] +1 zV[c,rt,Subscription,Linkset]
  +1 zV[c,lb,Ownership,Linkset] <= +2
 triangle2[Subscription,Ownership,Version,r]: +1 zV[r,lb,Subscription,Ownership] +1 zV[r,rt,Subscription,Version]
  +1 zV[r,lb,Ownership,Version] <= +2
 triangle2[Subscription,Ownership,Version,c]: +1 zV[c,lb,Subscription,Ownership] +1 zV[c,rt,Subscription,Version]
  +1 zV[c,lb,Ownership,Version] <= +2
 triangle2[Subscription,Ownership,Dependency,r]: +1 zV[r,lb,Subscription,Ownership] +1 zV[r,rt,Subscription,Dependency]
  +1 zV[r,lb,Ownership,Dependency] <= +2
 triangle2[Subscription,Ownership,Dependency,c]: +1 zV[c,lb,Subscription,Ownership] +1 zV[c,rt,Subscription,Dependency]
  +1 zV[c,lb,Ownership,Dependency] <= +2
 triangle2[Subscription,Linkset,Version,r]: +1 zV[r,lb,Subscription,Linkset] +1 zV[r,rt,Subscription,Version]
  +1 zV[r,lb,Linkset,Version] <= +2
 triangle2[Subscription,Linkset,Version,c]: +1 zV[c,lb,Subscription,Linkset] +1 zV[c,rt,Subscription,Version]
  +1 zV[c,lb,Linkset,Version] <= +2
 triangle2[Subscription,Linkset,Dependency,r]: +1 zV[r,lb,Subscription,Linkset] +1 zV[r,rt,Subscription,Dependency]
  +1 zV[r,lb,Linkset,Dependency] <= +2
 triangle2[Subscription,Linkset,Dependency,c]: +1 zV[c,lb,Subscription,Linkset] +1 zV[c,rt,Subscription,Dependency]
  +1 zV[c,lb,Linkset,Dependency] <= +2
 triangle2[Subscription,Version,Dependency,r]: +1 zV[r,lb,Subscription,Version] +1 zV[r,rt,Subscription,Dependency]
  +1 zV[r,lb,Version,Dependency] <= +2
 triangle2[Subscription,Version,Dependency,c]: +1 zV[c,lb,Subscription,Version] +1 zV[c,rt,Subscription,Dependency]
  +1 zV[c,lb,Version,Dependency] <= +2
 triangle2[Ownership,Linkset,Version,r]: +1 zV[r,lb,Ownership,Linkset] +1 zV[r,rt,Ownership,Version] +1 zV[r,lb,Linkset,Version]
  <= +2
 triangle2[Ownership,Linkset,Version,c]: +1 zV[c,lb,Ownership,Linkset] +1 zV[c,rt,Ownership,Version] +1 zV[c,lb,Linkset,Version]
  <= +2
 triangle2[Ownership,Linkset,Dependency,r]: +1 zV[r,lb,Ownership,Linkset] +1 zV[r,rt,Ownership,Dependency]
  +1 zV[r,lb,Linkset,Dependency] <= +2
 triangle2[Ownership,Linkset,Dependency,c]: +1 zV[c,lb,Ownership,Linkset] +1 zV[c,rt,Ownership,Dependency]
  +1 zV[c,lb,Linkset,Dependency] <= +2
 triangle2[Ownership,Version,Dependency,r]: +1 zV[r,lb,Ownership,Version] +1 zV[r,rt,Ownership,Dependency]
  +1 zV[r,lb,Version,Dependency] <= +2
 triangle2[Ownership,Version,Dependency,c]: +1 zV[c,lb,Ownership,Version] +1 zV[c,rt,Ownership,Dependency]
  +1 zV[c,lb,Version,Dependency] <= +2
 triangle2[Linkset,Version,Dependency,r]: +1 zV[r,lb,Linkset,Version] +1 zV[r,rt,Linkset,Dependency] +1 zV[r,lb,Version,Dependency]
  <= +2
 triangle2[Linkset,Version,Dependency,c]: +1 zV[c,lb,Linkset,Version] +1 zV[c,rt,Linkset,Dependency] +1 zV[c,lb,Version,Dependency]
  <= +2
Bounds
 0 <= zV[r,lb,Version,Dependency] <= 1
 0 <= zV[c,rt,Version,Dependency] <= 1
 0 <= zV[r,rt,Version,Dependency] <= 1
 0 <= zV[c,lb,Linkset,Dependency] <= 1
 0 <= zV[r,lb,Linkset,Dependency] <= 1
 0 <= zV[c,rt,Linkset,Dependency] <= 1
 0 <= zV[r,rt,Linkset,Dependency] <= 1
 0 <= zV[c,lb,Linkset,Version] <= 1
 0 <= zV[r,lb,Linkset,Version] <= 1
 0 <= zV[c,rt,Linkset,Version] <= 1
 0 <= zV[r,rt,Linkset,Version] <= 1
 0 <= zV[c,lb,Ownership,Dependency] <= 1
 0 <= zV[r,lb,Ownership,Dependency] <= 1
 0 <= zV[c,rt,Ownership,Dependency] <= 1
 0 <= zV[r,rt,Ownership,Dependency] <= 1
 0 <= zV[c,lb,Ownership,Version] <= 1
 0 <= zV[r,lb,Ownership,Version] <= 1
 0 <= zV[r,rt,Rubygem,User] <= 1
 0 <= zV[c,rt,Rubygem,User] <= 1
 0 <= zV[r,lb,Rubygem,User] <= 1
 0 <= zV[c,lb,Rubygem,User] <= 1
 0 <= zV[r,rt,Rubygem,WebHook] <= 1
 0 <= zV[c,rt,Rubygem,WebHook] <= 1
 0 <= zV[r,lb,Rubygem,WebHook] <= 1
 0 <= zV[c,lb,Rubygem,WebHook] <= 1
 0 <= zV[r,rt,Rubygem,Subscription] <= 1
 0 <= zV[c,rt,Rubygem,Subscription] <= 1
 0 <= zV[r,lb,Rubygem,Subscription] <= 1
 0 <= zV[c,lb,Rubygem,Subscription] <= 1
 0 <= zV[r,rt,Rubygem,Ownership] <= 1
 0 <= zV[c,rt,Rubygem,Ownership] <= 1
 0 <= zV[r,lb,Rubygem,Ownership] <= 1
 0 <= zV[c,lb,Rubygem,Ownership] <= 1
 0 <= zV[r,rt,Rubygem,Linkset] <= 1
 0 <= zV[c,rt,Rubygem,Linkset] <= 1
 0 <= zV[r,lb,Rubygem,Linkset] <= 1
 0 <= zV[c,lb,Rubygem,Linkset] <= 1
 0 <= zV[r,rt,Rubygem,Version] <= 1
 0 <= zV[c,rt,Rubygem,Version] <= 1
 0 <= zV[r,lb,Rubygem,Version] <= 1
 0 <= zV[c,lb,Rubygem,Version] <= 1
 0 <= zV[r,rt,Rubygem,Dependency] <= 1
 0 <= zV[c,rt,Rubygem,Dependency] <= 1
 0 <= zV[r,lb,Rubygem,Dependency] <= 1
 0 <= zV[c,lb,Rubygem,Dependency] <= 1
 0 <= zV[r,rt,User,WebHook] <= 1
 0 <= zV[c,rt,User,WebHook] <= 1
 0 <= zV[r,lb,User,WebHook] <= 1
 0 <= zV[c,lb,User,WebHook] <= 1
 0 <= zV[r,rt,User,Subscription] <= 1
 0 <= zV[c,rt,User,Subscription] <= 1
 0 <= zV[r,lb,User,Subscription] <= 1
 0 <= zV[c,lb,User,Subscription] <= 1
 0 <= zV[r,rt,User,Ownership] <= 1
 0 <= zV[c,rt,User,Ownership] <= 1
 0 <= zV[r,lb,User,Ownership] <= 1
 0 <= zV[c,lb,User,Ownership] <= 1
 0 <= zV[r,rt,User,Linkset] <= 1
 0 <= zV[c,rt,User,Linkset] <= 1
 0 <= zV[r,lb,User,Linkset] <= 1
 0 <= zV[c,lb,User,Linkset] <= 1
 0 <= zV[r,rt,User,Version] <= 1
 0 <= zV[c,rt,User,Version] <= 1
 0 <= zV[r,lb,User,Version] <= 1
 0 <= zV[c,lb,User,Version] <= 1
 0 <= zV[r,rt,User,Dependency] <= 1
 0 <= zV[c,rt,User,Dependency] <= 1
 0 <= zV[r,lb,User,Dependency] <= 1
 0 <= zV[c,lb,User,Dependency] <= 1
 0 <= zV[r,rt,WebHook,Subscription] <= 1
 0 <= zV[c,rt,WebHook,Subscription] <= 1
 0 <= zV[r,lb,WebHook,Subscription] <= 1
 0 <= zV[c,lb,WebHook,Subscription] <= 1
 0 <= zV[r,rt,WebHook,Ownership] <= 1
 0 <= zV[c,rt,WebHook,Ownership] <= 1
 0 <= zV[r,lb,WebHook,Ownership] <= 1
 0 <= zV[c,lb,WebHook,Ownership] <= 1
 0 <= zV[r,rt,WebHook,Linkset] <= 1
 0 <= zV[c,rt,WebHook,Linkset] <= 1
 0 <= zV[r,lb,WebHook,Linkset] <= 1
 0 <= zV[c,lb,WebHook,Linkset] <= 1
 0 <= zV[r,rt,WebHook,Version] <= 1
 0 <= zV[c,rt,WebHook,Version] <= 1
 0 <= zV[r,lb,WebHook,Version] <= 1
 0 <= zV[c,lb,WebHook,Version] <= 1
 0 <= zV[r,rt,WebHook,Dependency] <= 1
 0 <= zV[c,rt,WebHook,Dependency] <= 1
 0 <= zV[r,lb,WebHook,Dependency] <= 1
 0 <= zV[c,lb,WebHook,Dependency] <= 1
 0 <= zV[r,rt,Subscription,Ownership] <= 1
 0 <= zV[c,rt,Subscription,Ownership] <= 1
 0 <= zV[r,lb,Subscription,Ownership] <= 1
 0 <= zV[c,lb,Subscription,Ownership] <= 1
 0 <= zV[r,rt,Subscription,Linkset] <= 1
 0 <= zV[c,rt,Subscription,Linkset] <= 1
 0 <= zV[r,lb,Subscription,Linkset] <= 1
 0 <= zV[c,lb,Subscription,Linkset] <= 1
 0 <= zV[r,rt,Subscription,Version] <= 1
 0 <= zV[c,rt,Subscription,Version] <= 1
 0 <= zV[r,lb,Subscription,Version] <= 1
 0 <= zV[c,lb,Subscription,Version] <= 1
 0 <= zV[r,rt,Subscription,Dependency] <= 1
 0 <= zV[c,rt,Subscription,Dependency] <= 1
 0 <= zV[r,lb,Subscription,Dependency] <= 1
 0 <= zV[c,lb,Subscription,Dependency] <= 1
 0 <= zV[r,rt,Ownership,Linkset] <= 1
 0 <= zV[c,rt,Ownership,Linkset] <= 1
 0 <= zV[r,lb,Ownership,Linkset] <= 1
 0 <= zV[c,lb,Ownership,Linkset] <= 1
 0 <= zV[r,rt,Ownership,Version] <= 1
 0 <= zV[c,rt,Ownership,Version] <= 1
 0 <= zV[c,lb,Version,Dependency] <= 1
 1 <= x[c,Version] <= 20
 1 <= x[c,Linkset] <= 25
 1 <= x[c,Ownership] <= 23
 1 <= x[c,Subscription] <= 24
 1 <= x[c,WebHook] <= 21
 1 <= x[c,User] <= 18
 1 <= x[c,Rubygem] <= 23
 1 <= x[r,Dependency] <= 27
 1 <= x[r,Version] <= 23
 1 <= x[r,Linkset] <= 25
 1 <= x[r,Ownership] <= 27
 1 <= x[r,Subscription] <= 28
 1 <= x[r,WebHook] <= 27
 1 <= x[r,User] <= 23
 1 <= x[r,Rubygem] <= 27
 1 <= x[c,Dependency] <= 22
 0 <= dist[c,Rubygem,Dependency] <= 31.5
 0 <= dist[c,Version,Dependency] <= 32
 center_dist[r,Rubygem] free
 center_dist[r,User] free
 center_dist[r,WebHook] free
 center_dist[r,Subscription] free
 center_dist[r,Ownership] free
 center_dist[r,Linkset] free
 center_dist[r,Version] free
 center_dist[r,Dependency] free
 center_dist[c,Rubygem] free
 center_dist[c,User] free
 center_dist[c,WebHook] free
 center_dist[c,Subscription] free
 center_dist[c,Ownership] free
 center_dist[c,Linkset] free
 center_dist[c,Version] free
 center_dist[c,Dependency] free
 0 <= dist[r,Rubygem,User] <= 33
 0 <= dist[r,User,WebHook] <= 33
 0 <= dist[r,User,Subscription] <= 33.5
 0 <= dist[r,User,Ownership] <= 33
 0 <= dist[r,Rubygem,Subscription] <= 31.5
 0 <= dist[r,Rubygem,WebHook] <= 31
 0 <= dist[r,Rubygem,Linkset] <= 32
 0 <= dist[r,Rubygem,Ownership] <= 31
 0 <= dist[r,Rubygem,Version] <= 33
 0 <= dist[r,Rubygem,Dependency] <= 31
 0 <= dist[r,Version,Dependency] <= 33
 0 <= dist[c,Rubygem,User] <= 33.5
 0 <= dist[c,User,WebHook] <= 32.5
 0 <= dist[c,User,Subscription] <= 34
 0 <= dist[c,User,Ownership] <= 33.5
 0 <= dist[c,Rubygem,Subscription] <= 31.5
 0 <= dist[c,Rubygem,WebHook] <= 32
 0 <= dist[c,Rubygem,Linkset] <= 32
 0 <= dist[c,Rubygem,Ownership] <= 31
 0 <= dist[c,Rubygem,Version] <= 32.5
Binaries
 zV[r,lb,Version,Dependency] zV[c,rt,Version,Dependency] zV[r,rt,Version,Dependency] zV[c,lb,Linkset,Dependency]
 zV[r,lb,Linkset,Dependency] zV[c,rt,Linkset,Dependency] zV[r,rt,Linkset,Dependency] zV[c,lb,Linkset,Version]
 zV[r,lb,Linkset,Version] zV[c,rt,Linkset,Version] zV[r,rt,Linkset,Version] zV[c,lb,Ownership,Dependency]
 zV[r,lb,Ownership,Dependency] zV[c,rt,Ownership,Dependency] zV[r,rt,Ownership,Dependency] zV[c,lb,Ownership,Version]
 zV[r,lb,Ownership,Version] zV[r,rt,Rubygem,User] zV[c,rt,Rubygem,User] zV[r,lb,Rubygem,User] zV[c,lb,Rubygem,User]
 zV[r,rt,Rubygem,WebHook] zV[c,rt,Rubygem,WebHook] zV[r,lb,Rubygem,WebHook] zV[c,lb,Rubygem,WebHook] zV[r,rt,Rubygem,Subscription]
 zV[c,rt,Rubygem,Subscription] zV[r,lb,Rubygem,Subscription] zV[c,lb,Rubygem,Subscription] zV[r,rt,Rubygem,Ownership]
 zV[c,rt,Rubygem,Ownership] zV[r,lb,Rubygem,Ownership] zV[c,lb,Rubygem,Ownership] zV[r,rt,Rubygem,Linkset]
 zV[c,rt,Rubygem,Linkset] zV[r,lb,Rubygem,Linkset] zV[c,lb,Rubygem,Linkset] zV[r,rt,Rubygem,Version] zV[c,rt,Rubygem,Version]
 zV[r,lb,Rubygem,Version] zV[c,lb,Rubygem,Version] zV[r,rt,Rubygem,Dependency] zV[c,rt,Rubygem,Dependency]
 zV[r,lb,Rubygem,Dependency] zV[c,lb,Rubygem,Dependency] zV[r,rt,User,WebHook] zV[c,rt,User,WebHook] zV[r,lb,User,WebHook]
 zV[c,lb,User,WebHook] zV[r,rt,User,Subscription] zV[c,rt,User,Subscription] zV[r,lb,User,Subscription]
 zV[c,lb,User,Subscription] zV[r,rt,User,Ownership] zV[c,rt,User,Ownership] zV[r,lb,User,Ownership] zV[c,lb,User,Ownership]
 zV[r,rt,User,Linkset] zV[c,rt,User,Linkset] zV[r,lb,User,Linkset] zV[c,lb,User,Linkset] zV[r,rt,User,Version]
 zV[c,rt,User,Version] zV[r,lb,User,Version] zV[c,lb,User,Version] zV[r,rt,User,Dependency] zV[c,rt,User,Dependency]
 zV[r,lb,User,Dependency] zV[c,lb,User,Dependency] zV[r,rt,WebHook,Subscription] zV[c,rt,WebHook,Subscription]
 zV[r,lb,WebHook,Subscription] zV[c,lb,WebHook,Subscription] zV[r,rt,WebHook,Ownership] zV[c,rt,WebHook,Ownership]
 zV[r,lb,WebHook,Ownership] zV[c,lb,WebHook,Ownership] zV[r,rt,WebHook,Linkset] zV[c,rt,WebHook,Linkset]
 zV[r,lb,WebHook,Linkset] zV[c,lb,WebHook,Linkset] zV[r,rt,WebHook,Version] zV[c,rt,WebHook,Version] zV[r,lb,WebHook,Version]
 zV[c,lb,WebHook,Version] zV[r,rt,WebHook,Dependency] zV[c,rt,WebHook,Dependency] zV[r,lb,WebHook,Dependency]
 zV[c,lb,WebHook,Dependency] zV[r,rt,Subscription,Ownership] zV[c,rt,Subscription,Ownership] zV[r,lb,Subscription,Ownership]
 zV[c,lb,Subscription,Ownership] zV[r,rt,Subscription,Linkset] zV[c,rt,Subscription,Linkset] zV[r,lb,Subscription,Linkset]
 zV[c,lb,Subscription,Linkset] zV[r,rt,Subscription,Version] zV[c,rt,Subscription,Version] zV[r,lb,Subscription,Version]
 zV[c,lb,Subscription,Version] zV[r,rt,Subscription,Dependency] zV[c,rt,Subscription,Dependency] zV[r,lb,Subscription,Dependency]
 zV[c,lb,Subscription,Dependency] zV[r,rt,Ownership,Linkset] zV[c,rt,Ownership,Linkset] zV[r,lb,Ownership,Linkset]
 zV[c,lb,Ownership,Linkset] zV[r,rt,Ownership,Version] zV[c,rt,Ownership,Version] zV[c,lb,Version,Dependency]
Generals
 x[c,Version] x[c,Linkset] x[c,Ownership] x[c,Subscription] x[c,WebHook] x[c,User] x[c,Rubygem] x[r,Dependency]
 x[r,Version] x[r,Linkset] x[r,Ownership] x[r,Subscription] x[r,WebHook] x[r,User] x[r,Rubygem] x[c,Dependency]
End
