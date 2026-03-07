\ SCIP STATISTICS
\   Problem name     : graphdraw-domain
\   Variables        : 254 (180 binary, 20 integer, 0 implicit integer, 54 continuous)
\   Constraints      : 865
Minimize
 Obj: +1 center_dist[c,Therapist] +1 center_dist[c,Instructor] +86 dist[r,Client,CertificateTemplate]
      +86 dist[r,Client,CourseSession] +86 dist[r,Client,Instructor] +86 dist[r,Client,Location] +86 dist[r,Client,User]
      +86 dist[r,CertificateTemplate,CourseSession] +86 dist[r,Course,CourseSession] +86 dist[r,CourseSession,Instructor]
      +86 dist[r,Instructor,Location] +86 dist[r,Clinic,Location] +86 dist[r,Clinic,Therapist] +86 dist[r,Location,Therapist]
      +86 dist[r,Location,CourseSession] +86 dist[r,Therapist,CourseSession] +86 dist[r,Role,User] +86 dist[r,User,Instructor]
      +86 dist[r,User,Therapist] +86 dist[c,Client,CertificateTemplate] +86 dist[c,Client,CourseSession]
      +86 dist[c,Client,Instructor] +86 dist[c,Client,Location] +86 dist[c,Client,User] +86 dist[c,CertificateTemplate,CourseSession]
      +86 dist[c,Course,CourseSession] +86 dist[c,CourseSession,Instructor] +86 dist[c,Instructor,Location]
      +86 dist[c,Clinic,Location] +86 dist[c,Clinic,Therapist] +86 dist[c,Location,Therapist] +86 dist[c,Location,CourseSession]
      +86 dist[c,Therapist,CourseSession] +86 dist[c,Role,User] +86 dist[c,User,Instructor] +86 dist[c,User,Therapist]
      +1 center_dist[r,Client] +1 center_dist[r,CertificateTemplate] +1 center_dist[r,Course] +1 center_dist[r,Role]
      +1 center_dist[r,User] +1 center_dist[r,CourseSession] +1 center_dist[r,Clinic] +1 center_dist[r,Location]
      +1 center_dist[r,Therapist] +1 center_dist[r,Instructor] +1 center_dist[c,Client] +1 center_dist[c,CertificateTemplate]
      +1 center_dist[c,Course] +1 center_dist[c,Role] +1 center_dist[c,User] +1 center_dist[c,CourseSession]
      +1 center_dist[c,Clinic] +1 center_dist[c,Location]
Subject to
 choose1[Client,CertificateTemplate]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[c,rt,Client,CertificateTemplate]
  +1 zV[r,lb,Client,CertificateTemplate] +1 zV[c,lb,Client,CertificateTemplate] = +1
 choose1[Client,Course]: +1 zV[r,rt,Client,Course] +1 zV[c,rt,Client,Course] +1 zV[r,lb,Client,Course]
  +1 zV[c,lb,Client,Course] = +1
 choose1[Client,Role]: +1 zV[r,rt,Client,Role] +1 zV[c,rt,Client,Role] +1 zV[r,lb,Client,Role] +1 zV[c,lb,Client,Role]
  = +1
 choose1[Client,User]: +1 zV[r,rt,Client,User] +1 zV[c,rt,Client,User] +1 zV[r,lb,Client,User] +1 zV[c,lb,Client,User]
  = +1
 choose1[Client,CourseSession]: +1 zV[r,rt,Client,CourseSession] +1 zV[c,rt,Client,CourseSession] +1 zV[r,lb,Client,CourseSession]
  +1 zV[c,lb,Client,CourseSession] = +1
 choose1[Client,Clinic]: +1 zV[r,rt,Client,Clinic] +1 zV[c,rt,Client,Clinic] +1 zV[r,lb,Client,Clinic]
  +1 zV[c,lb,Client,Clinic] = +1
 choose1[Client,Location]: +1 zV[r,rt,Client,Location] +1 zV[c,rt,Client,Location] +1 zV[r,lb,Client,Location]
  +1 zV[c,lb,Client,Location] = +1
 choose1[Client,Therapist]: +1 zV[r,rt,Client,Therapist] +1 zV[c,rt,Client,Therapist] +1 zV[r,lb,Client,Therapist]
  +1 zV[c,lb,Client,Therapist] = +1
 choose1[Client,Instructor]: +1 zV[r,rt,Client,Instructor] +1 zV[c,rt,Client,Instructor] +1 zV[r,lb,Client,Instructor]
  +1 zV[c,lb,Client,Instructor] = +1
 choose1[CertificateTemplate,Course]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Course]
  +1 zV[r,lb,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Course] = +1
 choose1[CertificateTemplate,Role]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,Role]
  +1 zV[r,lb,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,Role] = +1
 choose1[CertificateTemplate,User]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,User]
  +1 zV[r,lb,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,User] = +1
 choose1[CertificateTemplate,CourseSession]: +1 zV[r,rt,CertificateTemplate,CourseSession] +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,CertificateTemplate,CourseSession] +1 zV[c,lb,CertificateTemplate,CourseSession] = +1
 choose1[CertificateTemplate,Clinic]: +1 zV[r,rt,CertificateTemplate,Clinic] +1 zV[c,rt,CertificateTemplate,Clinic]
  +1 zV[r,lb,CertificateTemplate,Clinic] +1 zV[c,lb,CertificateTemplate,Clinic] = +1
 choose1[CertificateTemplate,Location]: +1 zV[r,rt,CertificateTemplate,Location] +1 zV[c,rt,CertificateTemplate,Location]
  +1 zV[r,lb,CertificateTemplate,Location] +1 zV[c,lb,CertificateTemplate,Location] = +1
 choose1[CertificateTemplate,Therapist]: +1 zV[r,rt,CertificateTemplate,Therapist] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,CertificateTemplate,Therapist] +1 zV[c,lb,CertificateTemplate,Therapist] = +1
 choose1[CertificateTemplate,Instructor]: +1 zV[r,rt,CertificateTemplate,Instructor] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,CertificateTemplate,Instructor] +1 zV[c,lb,CertificateTemplate,Instructor] = +1
 choose1[Course,Role]: +1 zV[r,rt,Course,Role] +1 zV[c,rt,Course,Role] +1 zV[r,lb,Course,Role] +1 zV[c,lb,Course,Role]
  = +1
 choose1[Course,User]: +1 zV[r,rt,Course,User] +1 zV[c,rt,Course,User] +1 zV[r,lb,Course,User] +1 zV[c,lb,Course,User]
  = +1
 choose1[Course,CourseSession]: +1 zV[r,rt,Course,CourseSession] +1 zV[c,rt,Course,CourseSession] +1 zV[r,lb,Course,CourseSession]
  +1 zV[c,lb,Course,CourseSession] = +1
 choose1[Course,Clinic]: +1 zV[r,rt,Course,Clinic] +1 zV[c,rt,Course,Clinic] +1 zV[r,lb,Course,Clinic]
  +1 zV[c,lb,Course,Clinic] = +1
 choose1[Course,Location]: +1 zV[r,rt,Course,Location] +1 zV[c,rt,Course,Location] +1 zV[r,lb,Course,Location]
  +1 zV[c,lb,Course,Location] = +1
 choose1[Course,Therapist]: +1 zV[r,rt,Course,Therapist] +1 zV[c,rt,Course,Therapist] +1 zV[r,lb,Course,Therapist]
  +1 zV[c,lb,Course,Therapist] = +1
 choose1[Course,Instructor]: +1 zV[r,rt,Course,Instructor] +1 zV[c,rt,Course,Instructor] +1 zV[r,lb,Course,Instructor]
  +1 zV[c,lb,Course,Instructor] = +1
 choose1[Role,User]: +1 zV[r,rt,Role,User] +1 zV[c,rt,Role,User] +1 zV[r,lb,Role,User] +1 zV[c,lb,Role,User]
  = +1
 choose1[Role,CourseSession]: +1 zV[r,rt,Role,CourseSession] +1 zV[c,rt,Role,CourseSession] +1 zV[r,lb,Role,CourseSession]
  +1 zV[c,lb,Role,CourseSession] = +1
 choose1[Role,Clinic]: +1 zV[r,rt,Role,Clinic] +1 zV[c,rt,Role,Clinic] +1 zV[r,lb,Role,Clinic] +1 zV[c,lb,Role,Clinic]
  = +1
 choose1[Role,Location]: +1 zV[r,rt,Role,Location] +1 zV[c,rt,Role,Location] +1 zV[r,lb,Role,Location]
  +1 zV[c,lb,Role,Location] = +1
 choose1[Role,Therapist]: +1 zV[r,rt,Role,Therapist] +1 zV[c,rt,Role,Therapist] +1 zV[r,lb,Role,Therapist]
  +1 zV[c,lb,Role,Therapist] = +1
 choose1[Role,Instructor]: +1 zV[r,rt,Role,Instructor] +1 zV[c,rt,Role,Instructor] +1 zV[r,lb,Role,Instructor]
  +1 zV[c,lb,Role,Instructor] = +1
 choose1[User,CourseSession]: +1 zV[r,rt,User,CourseSession] +1 zV[c,rt,User,CourseSession] +1 zV[r,lb,User,CourseSession]
  +1 zV[c,lb,User,CourseSession] = +1
 choose1[User,Clinic]: +1 zV[r,rt,User,Clinic] +1 zV[c,rt,User,Clinic] +1 zV[r,lb,User,Clinic] +1 zV[c,lb,User,Clinic]
  = +1
 choose1[User,Location]: +1 zV[r,rt,User,Location] +1 zV[c,rt,User,Location] +1 zV[r,lb,User,Location]
  +1 zV[c,lb,User,Location] = +1
 choose1[User,Therapist]: +1 zV[r,rt,User,Therapist] +1 zV[c,rt,User,Therapist] +1 zV[r,lb,User,Therapist]
  +1 zV[c,lb,User,Therapist] = +1
 choose1[User,Instructor]: +1 zV[r,rt,User,Instructor] +1 zV[c,rt,User,Instructor] +1 zV[r,lb,User,Instructor]
  +1 zV[c,lb,User,Instructor] = +1
 choose1[CourseSession,Clinic]: +1 zV[r,rt,CourseSession,Clinic] +1 zV[c,rt,CourseSession,Clinic] +1 zV[r,lb,CourseSession,Clinic]
  +1 zV[c,lb,CourseSession,Clinic] = +1
 choose1[CourseSession,Location]: +1 zV[r,rt,CourseSession,Location] +1 zV[c,rt,CourseSession,Location]
  +1 zV[r,lb,CourseSession,Location] +1 zV[c,lb,CourseSession,Location] = +1
 choose1[CourseSession,Therapist]: +1 zV[r,rt,CourseSession,Therapist] +1 zV[c,rt,CourseSession,Therapist]
  +1 zV[r,lb,CourseSession,Therapist] +1 zV[c,lb,CourseSession,Therapist] = +1
 choose1[CourseSession,Instructor]: +1 zV[r,rt,CourseSession,Instructor] +1 zV[c,rt,CourseSession,Instructor]
  +1 zV[r,lb,CourseSession,Instructor] +1 zV[c,lb,CourseSession,Instructor] = +1
 choose1[Clinic,Location]: +1 zV[r,rt,Clinic,Location] +1 zV[c,rt,Clinic,Location] +1 zV[r,lb,Clinic,Location]
  +1 zV[c,lb,Clinic,Location] = +1
 choose1[Clinic,Therapist]: +1 zV[r,rt,Clinic,Therapist] +1 zV[c,rt,Clinic,Therapist] +1 zV[r,lb,Clinic,Therapist]
  +1 zV[c,lb,Clinic,Therapist] = +1
 choose1[Clinic,Instructor]: +1 zV[r,rt,Clinic,Instructor] +1 zV[c,rt,Clinic,Instructor] +1 zV[r,lb,Clinic,Instructor]
  +1 zV[c,lb,Clinic,Instructor] = +1
 choose1[Location,Therapist]: +1 zV[r,rt,Location,Therapist] +1 zV[c,rt,Location,Therapist] +1 zV[r,lb,Location,Therapist]
  +1 zV[c,lb,Location,Therapist] = +1
 choose1[Location,Instructor]: +1 zV[r,rt,Location,Instructor] +1 zV[c,rt,Location,Instructor] +1 zV[r,lb,Location,Instructor]
  +1 zV[c,lb,Location,Instructor] = +1
 choose1[Therapist,Instructor]: +1 zV[r,rt,Therapist,Instructor] +1 zV[c,rt,Therapist,Instructor] +1 zV[r,lb,Therapist,Instructor]
  +1 zV[c,lb,Therapist,Instructor] = +1
 centerDistAxis1[r,Client]: +1 x[r,Client] +1 center_dist[r,Client] >= +17.5
 centerDistAxis1[r,CertificateTemplate]: +1 x[r,CertificateTemplate] +1 center_dist[r,CertificateTemplate]
  >= +18.5
 centerDistAxis1[r,Course]: +1 x[r,Course] +1 center_dist[r,Course] >= +19
 centerDistAxis1[r,Role]: +1 x[r,Role] +1 center_dist[r,Role] >= +19
 centerDistAxis1[r,User]: +1 x[r,User] +1 center_dist[r,User] >= +18.5
 centerDistAxis1[r,CourseSession]: +1 x[r,CourseSession] +1 center_dist[r,CourseSession] >= +17.5
 centerDistAxis1[r,Clinic]: +1 x[r,Clinic] +1 center_dist[r,Clinic] >= +18
 centerDistAxis1[r,Location]: +1 x[r,Location] +1 center_dist[r,Location] >= +17.5
 centerDistAxis1[r,Therapist]: +1 x[r,Therapist] +1 center_dist[r,Therapist] >= +17
 centerDistAxis1[r,Instructor]: +1 x[r,Instructor] +1 center_dist[r,Instructor] >= +17
 centerDistAxis1[c,Client]: +1 x[c,Client] +1 center_dist[c,Client] >= +16.5
 centerDistAxis1[c,CertificateTemplate]: +1 x[c,CertificateTemplate] +1 center_dist[c,CertificateTemplate]
  >= +16
 centerDistAxis1[c,Course]: +1 x[c,Course] +1 center_dist[c,Course] >= +17
 centerDistAxis1[c,Role]: +1 x[c,Role] +1 center_dist[c,Role] >= +17
 centerDistAxis1[c,User]: +1 x[c,User] +1 center_dist[c,User] >= +16.5
 centerDistAxis1[c,CourseSession]: +1 x[c,CourseSession] +1 center_dist[c,CourseSession] >= +15
 centerDistAxis1[c,Clinic]: +1 x[c,Clinic] +1 center_dist[c,Clinic] >= +16.5
 centerDistAxis1[c,Location]: +1 x[c,Location] +1 center_dist[c,Location] >= +16.5
 centerDistAxis1[c,Therapist]: +1 x[c,Therapist] +1 center_dist[c,Therapist] >= +16.5
 centerDistAxis1[c,Instructor]: +1 x[c,Instructor] +1 center_dist[c,Instructor] >= +16
 centerDistAxis2[r,Client]: -1 x[r,Client] +1 center_dist[r,Client] >= -17.5
 centerDistAxis2[r,CertificateTemplate]: -1 x[r,CertificateTemplate] +1 center_dist[r,CertificateTemplate]
  >= -18.5
 centerDistAxis2[r,Course]: -1 x[r,Course] +1 center_dist[r,Course] >= -19
 centerDistAxis2[r,Role]: -1 x[r,Role] +1 center_dist[r,Role] >= -19
 centerDistAxis2[r,User]: -1 x[r,User] +1 center_dist[r,User] >= -18.5
 centerDistAxis2[r,CourseSession]: -1 x[r,CourseSession] +1 center_dist[r,CourseSession] >= -17.5
 centerDistAxis2[r,Clinic]: -1 x[r,Clinic] +1 center_dist[r,Clinic] >= -18
 centerDistAxis2[r,Location]: -1 x[r,Location] +1 center_dist[r,Location] >= -17.5
 centerDistAxis2[r,Therapist]: -1 x[r,Therapist] +1 center_dist[r,Therapist] >= -17
 centerDistAxis2[r,Instructor]: -1 x[r,Instructor] +1 center_dist[r,Instructor] >= -17
 centerDistAxis2[c,Client]: -1 x[c,Client] +1 center_dist[c,Client] >= -16.5
 centerDistAxis2[c,CertificateTemplate]: -1 x[c,CertificateTemplate] +1 center_dist[c,CertificateTemplate]
  >= -16
 centerDistAxis2[c,Course]: -1 x[c,Course] +1 center_dist[c,Course] >= -17
 centerDistAxis2[c,Role]: -1 x[c,Role] +1 center_dist[c,Role] >= -17
 centerDistAxis2[c,User]: -1 x[c,User] +1 center_dist[c,User] >= -16.5
 centerDistAxis2[c,CourseSession]: -1 x[c,CourseSession] +1 center_dist[c,CourseSession] >= -15
 centerDistAxis2[c,Clinic]: -1 x[c,Clinic] +1 center_dist[c,Clinic] >= -16.5
 centerDistAxis2[c,Location]: -1 x[c,Location] +1 center_dist[c,Location] >= -16.5
 centerDistAxis2[c,Therapist]: -1 x[c,Therapist] +1 center_dist[c,Therapist] >= -16.5
 centerDistAxis2[c,Instructor]: -1 x[c,Instructor] +1 center_dist[c,Instructor] >= -16
 axisVdifLB[r,Client,CertificateTemplate]: -1 x[r,Client] +1 x[r,CertificateTemplate] -51 zV[r,lb,Client,CertificateTemplate]
  >= -42
 axisVdifLB[r,Client,Course]: -1 x[r,Client] +1 x[r,Course] -51 zV[r,lb,Client,Course] >= -42
 axisVdifLB[r,Client,Role]: -1 x[r,Client] +1 x[r,Role] -51 zV[r,lb,Client,Role] >= -42
 axisVdifLB[r,Client,User]: -1 x[r,Client] +1 x[r,User] -51 zV[r,lb,Client,User] >= -42
 axisVdifLB[r,Client,CourseSession]: -1 x[r,Client] +1 x[r,CourseSession] -51 zV[r,lb,Client,CourseSession]
  >= -42
 axisVdifLB[r,Client,Clinic]: -1 x[r,Client] +1 x[r,Clinic] -51 zV[r,lb,Client,Clinic] >= -42
 axisVdifLB[r,Client,Location]: -1 x[r,Client] +1 x[r,Location] -51 zV[r,lb,Client,Location] >= -42
 axisVdifLB[r,Client,Therapist]: -1 x[r,Client] +1 x[r,Therapist] -51 zV[r,lb,Client,Therapist] >= -42
 axisVdifLB[r,Client,Instructor]: -1 x[r,Client] +1 x[r,Instructor] -51 zV[r,lb,Client,Instructor] >= -42
 axisVdifLB[r,CertificateTemplate,Course]: -1 x[r,CertificateTemplate] +1 x[r,Course] -49 zV[r,lb,CertificateTemplate,Course]
  >= -42
 axisVdifLB[r,CertificateTemplate,Role]: -1 x[r,CertificateTemplate] +1 x[r,Role] -49 zV[r,lb,CertificateTemplate,Role]
  >= -42
 axisVdifLB[r,CertificateTemplate,User]: -1 x[r,CertificateTemplate] +1 x[r,User] -49 zV[r,lb,CertificateTemplate,User]
  >= -42
 axisVdifLB[r,CertificateTemplate,CourseSession]: -1 x[r,CertificateTemplate] +1 x[r,CourseSession] -49 zV[r,lb,CertificateTemplate,CourseSession]
  >= -42
 axisVdifLB[r,CertificateTemplate,Clinic]: -1 x[r,CertificateTemplate] +1 x[r,Clinic] -49 zV[r,lb,CertificateTemplate,Clinic]
  >= -42
 axisVdifLB[r,CertificateTemplate,Location]: -1 x[r,CertificateTemplate] +1 x[r,Location] -49 zV[r,lb,CertificateTemplate,Location]
  >= -42
 axisVdifLB[r,CertificateTemplate,Therapist]: -1 x[r,CertificateTemplate] +1 x[r,Therapist] -49 zV[r,lb,CertificateTemplate,Therapist]
  >= -42
 axisVdifLB[r,CertificateTemplate,Instructor]: -1 x[r,CertificateTemplate] +1 x[r,Instructor] -49 zV[r,lb,CertificateTemplate,Instructor]
  >= -42
 axisVdifLB[r,Course,Role]: -1 x[r,Course] +1 x[r,Role] -48 zV[r,lb,Course,Role] >= -42
 axisVdifLB[r,Course,User]: -1 x[r,Course] +1 x[r,User] -48 zV[r,lb,Course,User] >= -42
 axisVdifLB[r,Course,CourseSession]: -1 x[r,Course] +1 x[r,CourseSession] -48 zV[r,lb,Course,CourseSession]
  >= -42
 axisVdifLB[r,Course,Clinic]: -1 x[r,Course] +1 x[r,Clinic] -48 zV[r,lb,Course,Clinic] >= -42
 axisVdifLB[r,Course,Location]: -1 x[r,Course] +1 x[r,Location] -48 zV[r,lb,Course,Location] >= -42
 axisVdifLB[r,Course,Therapist]: -1 x[r,Course] +1 x[r,Therapist] -48 zV[r,lb,Course,Therapist] >= -42
 axisVdifLB[r,Course,Instructor]: -1 x[r,Course] +1 x[r,Instructor] -48 zV[r,lb,Course,Instructor] >= -42
 axisVdifLB[r,Role,User]: -1 x[r,Role] +1 x[r,User] -48 zV[r,lb,Role,User] >= -42
 axisVdifLB[r,Role,CourseSession]: -1 x[r,Role] +1 x[r,CourseSession] -48 zV[r,lb,Role,CourseSession]
  >= -42
 axisVdifLB[r,Role,Clinic]: -1 x[r,Role] +1 x[r,Clinic] -48 zV[r,lb,Role,Clinic] >= -42
 axisVdifLB[r,Role,Location]: -1 x[r,Role] +1 x[r,Location] -48 zV[r,lb,Role,Location] >= -42
 axisVdifLB[r,Role,Therapist]: -1 x[r,Role] +1 x[r,Therapist] -48 zV[r,lb,Role,Therapist] >= -42
 axisVdifLB[r,Role,Instructor]: -1 x[r,Role] +1 x[r,Instructor] -48 zV[r,lb,Role,Instructor] >= -42
 axisVdifLB[r,User,CourseSession]: -1 x[r,User] +1 x[r,CourseSession] -49 zV[r,lb,User,CourseSession]
  >= -42
 axisVdifLB[r,User,Clinic]: -1 x[r,User] +1 x[r,Clinic] -49 zV[r,lb,User,Clinic] >= -42
 axisVdifLB[r,User,Location]: -1 x[r,User] +1 x[r,Location] -49 zV[r,lb,User,Location] >= -42
 axisVdifLB[r,User,Therapist]: -1 x[r,User] +1 x[r,Therapist] -49 zV[r,lb,User,Therapist] >= -42
 axisVdifLB[r,User,Instructor]: -1 x[r,User] +1 x[r,Instructor] -49 zV[r,lb,User,Instructor] >= -42
 axisVdifLB[r,CourseSession,Clinic]: -1 x[r,CourseSession] +1 x[r,Clinic] -51 zV[r,lb,CourseSession,Clinic]
  >= -42
 axisVdifLB[r,CourseSession,Location]: -1 x[r,CourseSession] +1 x[r,Location] -51 zV[r,lb,CourseSession,Location]
  >= -42
 axisVdifLB[r,CourseSession,Therapist]: -1 x[r,CourseSession] +1 x[r,Therapist] -51 zV[r,lb,CourseSession,Therapist]
  >= -42
 axisVdifLB[r,CourseSession,Instructor]: -1 x[r,CourseSession] +1 x[r,Instructor] -51 zV[r,lb,CourseSession,Instructor]
  >= -42
 axisVdifLB[r,Clinic,Location]: -1 x[r,Clinic] +1 x[r,Location] -50 zV[r,lb,Clinic,Location] >= -42
 axisVdifLB[r,Clinic,Therapist]: -1 x[r,Clinic] +1 x[r,Therapist] -50 zV[r,lb,Clinic,Therapist] >= -42
 axisVdifLB[r,Clinic,Instructor]: -1 x[r,Clinic] +1 x[r,Instructor] -50 zV[r,lb,Clinic,Instructor] >= -42
 axisVdifLB[r,Location,Therapist]: -1 x[r,Location] +1 x[r,Therapist] -51 zV[r,lb,Location,Therapist]
  >= -42
 axisVdifLB[r,Location,Instructor]: -1 x[r,Location] +1 x[r,Instructor] -51 zV[r,lb,Location,Instructor]
  >= -42
 axisVdifLB[r,Therapist,Instructor]: -1 x[r,Therapist] +1 x[r,Instructor] -52 zV[r,lb,Therapist,Instructor]
  >= -42
 axisVdifLB[c,Client,CertificateTemplate]: -1 x[c,Client] +1 x[c,CertificateTemplate] -53 zV[c,lb,Client,CertificateTemplate]
  >= -42
 axisVdifLB[c,Client,Course]: -1 x[c,Client] +1 x[c,Course] -53 zV[c,lb,Client,Course] >= -42
 axisVdifLB[c,Client,Role]: -1 x[c,Client] +1 x[c,Role] -53 zV[c,lb,Client,Role] >= -42
 axisVdifLB[c,Client,User]: -1 x[c,Client] +1 x[c,User] -53 zV[c,lb,Client,User] >= -42
 axisVdifLB[c,Client,CourseSession]: -1 x[c,Client] +1 x[c,CourseSession] -53 zV[c,lb,Client,CourseSession]
  >= -42
 axisVdifLB[c,Client,Clinic]: -1 x[c,Client] +1 x[c,Clinic] -53 zV[c,lb,Client,Clinic] >= -42
 axisVdifLB[c,Client,Location]: -1 x[c,Client] +1 x[c,Location] -53 zV[c,lb,Client,Location] >= -42
 axisVdifLB[c,Client,Therapist]: -1 x[c,Client] +1 x[c,Therapist] -53 zV[c,lb,Client,Therapist] >= -42
 axisVdifLB[c,Client,Instructor]: -1 x[c,Client] +1 x[c,Instructor] -53 zV[c,lb,Client,Instructor] >= -42
 axisVdifLB[c,CertificateTemplate,Course]: -1 x[c,CertificateTemplate] +1 x[c,Course] -54 zV[c,lb,CertificateTemplate,Course]
  >= -42
 axisVdifLB[c,CertificateTemplate,Role]: -1 x[c,CertificateTemplate] +1 x[c,Role] -54 zV[c,lb,CertificateTemplate,Role]
  >= -42
 axisVdifLB[c,CertificateTemplate,User]: -1 x[c,CertificateTemplate] +1 x[c,User] -54 zV[c,lb,CertificateTemplate,User]
  >= -42
 axisVdifLB[c,CertificateTemplate,CourseSession]: -1 x[c,CertificateTemplate] +1 x[c,CourseSession] -54 zV[c,lb,CertificateTemplate,CourseSession]
  >= -42
 axisVdifLB[c,CertificateTemplate,Clinic]: -1 x[c,CertificateTemplate] +1 x[c,Clinic] -54 zV[c,lb,CertificateTemplate,Clinic]
  >= -42
 axisVdifLB[c,CertificateTemplate,Location]: -1 x[c,CertificateTemplate] +1 x[c,Location] -54 zV[c,lb,CertificateTemplate,Location]
  >= -42
 axisVdifLB[c,CertificateTemplate,Therapist]: -1 x[c,CertificateTemplate] +1 x[c,Therapist] -54 zV[c,lb,CertificateTemplate,Therapist]
  >= -42
 axisVdifLB[c,CertificateTemplate,Instructor]: -1 x[c,CertificateTemplate] +1 x[c,Instructor] -54 zV[c,lb,CertificateTemplate,Instructor]
  >= -42
 axisVdifLB[c,Course,Role]: -1 x[c,Course] +1 x[c,Role] -52 zV[c,lb,Course,Role] >= -42
 axisVdifLB[c,Course,User]: -1 x[c,Course] +1 x[c,User] -52 zV[c,lb,Course,User] >= -42
 axisVdifLB[c,Course,CourseSession]: -1 x[c,Course] +1 x[c,CourseSession] -52 zV[c,lb,Course,CourseSession]
  >= -42
 axisVdifLB[c,Course,Clinic]: -1 x[c,Course] +1 x[c,Clinic] -52 zV[c,lb,Course,Clinic] >= -42
 axisVdifLB[c,Course,Location]: -1 x[c,Course] +1 x[c,Location] -52 zV[c,lb,Course,Location] >= -42
 axisVdifLB[c,Course,Therapist]: -1 x[c,Course] +1 x[c,Therapist] -52 zV[c,lb,Course,Therapist] >= -42
 axisVdifLB[c,Course,Instructor]: -1 x[c,Course] +1 x[c,Instructor] -52 zV[c,lb,Course,Instructor] >= -42
 axisVdifLB[c,Role,User]: -1 x[c,Role] +1 x[c,User] -52 zV[c,lb,Role,User] >= -42
 axisVdifLB[c,Role,CourseSession]: -1 x[c,Role] +1 x[c,CourseSession] -52 zV[c,lb,Role,CourseSession]
  >= -42
 axisVdifLB[c,Role,Clinic]: -1 x[c,Role] +1 x[c,Clinic] -52 zV[c,lb,Role,Clinic] >= -42
 axisVdifLB[c,Role,Location]: -1 x[c,Role] +1 x[c,Location] -52 zV[c,lb,Role,Location] >= -42
 axisVdifLB[c,Role,Therapist]: -1 x[c,Role] +1 x[c,Therapist] -52 zV[c,lb,Role,Therapist] >= -42
 axisVdifLB[c,Role,Instructor]: -1 x[c,Role] +1 x[c,Instructor] -52 zV[c,lb,Role,Instructor] >= -42
 axisVdifLB[c,User,CourseSession]: -1 x[c,User] +1 x[c,CourseSession] -53 zV[c,lb,User,CourseSession]
  >= -42
 axisVdifLB[c,User,Clinic]: -1 x[c,User] +1 x[c,Clinic] -53 zV[c,lb,User,Clinic] >= -42
 axisVdifLB[c,User,Location]: -1 x[c,User] +1 x[c,Location] -53 zV[c,lb,User,Location] >= -42
 axisVdifLB[c,User,Therapist]: -1 x[c,User] +1 x[c,Therapist] -53 zV[c,lb,User,Therapist] >= -42
 axisVdifLB[c,User,Instructor]: -1 x[c,User] +1 x[c,Instructor] -53 zV[c,lb,User,Instructor] >= -42
 axisVdifLB[c,CourseSession,Clinic]: -1 x[c,CourseSession] +1 x[c,Clinic] -56 zV[c,lb,CourseSession,Clinic]
  >= -42
 axisVdifLB[c,CourseSession,Location]: -1 x[c,CourseSession] +1 x[c,Location] -56 zV[c,lb,CourseSession,Location]
  >= -42
 axisVdifLB[c,CourseSession,Therapist]: -1 x[c,CourseSession] +1 x[c,Therapist] -56 zV[c,lb,CourseSession,Therapist]
  >= -42
 axisVdifLB[c,CourseSession,Instructor]: -1 x[c,CourseSession] +1 x[c,Instructor] -56 zV[c,lb,CourseSession,Instructor]
  >= -42
 axisVdifLB[c,Clinic,Location]: -1 x[c,Clinic] +1 x[c,Location] -53 zV[c,lb,Clinic,Location] >= -42
 axisVdifLB[c,Clinic,Therapist]: -1 x[c,Clinic] +1 x[c,Therapist] -53 zV[c,lb,Clinic,Therapist] >= -42
 axisVdifLB[c,Clinic,Instructor]: -1 x[c,Clinic] +1 x[c,Instructor] -53 zV[c,lb,Clinic,Instructor] >= -42
 axisVdifLB[c,Location,Therapist]: -1 x[c,Location] +1 x[c,Therapist] -53 zV[c,lb,Location,Therapist]
  >= -42
 axisVdifLB[c,Location,Instructor]: -1 x[c,Location] +1 x[c,Instructor] -53 zV[c,lb,Location,Instructor]
  >= -42
 axisVdifLB[c,Therapist,Instructor]: -1 x[c,Therapist] +1 x[c,Instructor] -53 zV[c,lb,Therapist,Instructor]
  >= -42
 axisVdifRT[r,Client,CertificateTemplate]: +1 x[r,Client] -1 x[r,CertificateTemplate] -49 zV[r,rt,Client,CertificateTemplate]
  >= -42
 axisVdifRT[r,Client,Course]: +1 x[r,Client] -1 x[r,Course] -48 zV[r,rt,Client,Course] >= -42
 axisVdifRT[r,Client,Role]: +1 x[r,Client] -1 x[r,Role] -48 zV[r,rt,Client,Role] >= -42
 axisVdifRT[r,Client,User]: +1 x[r,Client] -1 x[r,User] -49 zV[r,rt,Client,User] >= -42
 axisVdifRT[r,Client,CourseSession]: +1 x[r,Client] -1 x[r,CourseSession] -51 zV[r,rt,Client,CourseSession]
  >= -42
 axisVdifRT[r,Client,Clinic]: +1 x[r,Client] -1 x[r,Clinic] -50 zV[r,rt,Client,Clinic] >= -42
 axisVdifRT[r,Client,Location]: +1 x[r,Client] -1 x[r,Location] -51 zV[r,rt,Client,Location] >= -42
 axisVdifRT[r,Client,Therapist]: +1 x[r,Client] -1 x[r,Therapist] -52 zV[r,rt,Client,Therapist] >= -42
 axisVdifRT[r,Client,Instructor]: +1 x[r,Client] -1 x[r,Instructor] -52 zV[r,rt,Client,Instructor] >= -42
 axisVdifRT[r,CertificateTemplate,Course]: +1 x[r,CertificateTemplate] -1 x[r,Course] -48 zV[r,rt,CertificateTemplate,Course]
  >= -42
 axisVdifRT[r,CertificateTemplate,Role]: +1 x[r,CertificateTemplate] -1 x[r,Role] -48 zV[r,rt,CertificateTemplate,Role]
  >= -42
 axisVdifRT[r,CertificateTemplate,User]: +1 x[r,CertificateTemplate] -1 x[r,User] -49 zV[r,rt,CertificateTemplate,User]
  >= -42
 axisVdifRT[r,CertificateTemplate,CourseSession]: +1 x[r,CertificateTemplate] -1 x[r,CourseSession] -51 zV[r,rt,CertificateTemplate,CourseSession]
  >= -42
 axisVdifRT[r,CertificateTemplate,Clinic]: +1 x[r,CertificateTemplate] -1 x[r,Clinic] -50 zV[r,rt,CertificateTemplate,Clinic]
  >= -42
 axisVdifRT[r,CertificateTemplate,Location]: +1 x[r,CertificateTemplate] -1 x[r,Location] -51 zV[r,rt,CertificateTemplate,Location]
  >= -42
 axisVdifRT[r,CertificateTemplate,Therapist]: +1 x[r,CertificateTemplate] -1 x[r,Therapist] -52 zV[r,rt,CertificateTemplate,Therapist]
  >= -42
 axisVdifRT[r,CertificateTemplate,Instructor]: +1 x[r,CertificateTemplate] -1 x[r,Instructor] -52 zV[r,rt,CertificateTemplate,Instructor]
  >= -42
 axisVdifRT[r,Course,Role]: +1 x[r,Course] -1 x[r,Role] -48 zV[r,rt,Course,Role] >= -42
 axisVdifRT[r,Course,User]: +1 x[r,Course] -1 x[r,User] -49 zV[r,rt,Course,User] >= -42
 axisVdifRT[r,Course,CourseSession]: +1 x[r,Course] -1 x[r,CourseSession] -51 zV[r,rt,Course,CourseSession]
  >= -42
 axisVdifRT[r,Course,Clinic]: +1 x[r,Course] -1 x[r,Clinic] -50 zV[r,rt,Course,Clinic] >= -42
 axisVdifRT[r,Course,Location]: +1 x[r,Course] -1 x[r,Location] -51 zV[r,rt,Course,Location] >= -42
 axisVdifRT[r,Course,Therapist]: +1 x[r,Course] -1 x[r,Therapist] -52 zV[r,rt,Course,Therapist] >= -42
 axisVdifRT[r,Course,Instructor]: +1 x[r,Course] -1 x[r,Instructor] -52 zV[r,rt,Course,Instructor] >= -42
 axisVdifRT[r,Role,User]: +1 x[r,Role] -1 x[r,User] -49 zV[r,rt,Role,User] >= -42
 axisVdifRT[r,Role,CourseSession]: +1 x[r,Role] -1 x[r,CourseSession] -51 zV[r,rt,Role,CourseSession]
  >= -42
 axisVdifRT[r,Role,Clinic]: +1 x[r,Role] -1 x[r,Clinic] -50 zV[r,rt,Role,Clinic] >= -42
 axisVdifRT[r,Role,Location]: +1 x[r,Role] -1 x[r,Location] -51 zV[r,rt,Role,Location] >= -42
 axisVdifRT[r,Role,Therapist]: +1 x[r,Role] -1 x[r,Therapist] -52 zV[r,rt,Role,Therapist] >= -42
 axisVdifRT[r,Role,Instructor]: +1 x[r,Role] -1 x[r,Instructor] -52 zV[r,rt,Role,Instructor] >= -42
 axisVdifRT[r,User,CourseSession]: +1 x[r,User] -1 x[r,CourseSession] -51 zV[r,rt,User,CourseSession]
  >= -42
 axisVdifRT[r,User,Clinic]: +1 x[r,User] -1 x[r,Clinic] -50 zV[r,rt,User,Clinic] >= -42
 axisVdifRT[r,User,Location]: +1 x[r,User] -1 x[r,Location] -51 zV[r,rt,User,Location] >= -42
 axisVdifRT[r,User,Therapist]: +1 x[r,User] -1 x[r,Therapist] -52 zV[r,rt,User,Therapist] >= -42
 axisVdifRT[r,User,Instructor]: +1 x[r,User] -1 x[r,Instructor] -52 zV[r,rt,User,Instructor] >= -42
 axisVdifRT[r,CourseSession,Clinic]: +1 x[r,CourseSession] -1 x[r,Clinic] -50 zV[r,rt,CourseSession,Clinic]
  >= -42
 axisVdifRT[r,CourseSession,Location]: +1 x[r,CourseSession] -1 x[r,Location] -51 zV[r,rt,CourseSession,Location]
  >= -42
 axisVdifRT[r,CourseSession,Therapist]: +1 x[r,CourseSession] -1 x[r,Therapist] -52 zV[r,rt,CourseSession,Therapist]
  >= -42
 axisVdifRT[r,CourseSession,Instructor]: +1 x[r,CourseSession] -1 x[r,Instructor] -52 zV[r,rt,CourseSession,Instructor]
  >= -42
 axisVdifRT[r,Clinic,Location]: +1 x[r,Clinic] -1 x[r,Location] -51 zV[r,rt,Clinic,Location] >= -42
 axisVdifRT[r,Clinic,Therapist]: +1 x[r,Clinic] -1 x[r,Therapist] -52 zV[r,rt,Clinic,Therapist] >= -42
 axisVdifRT[r,Clinic,Instructor]: +1 x[r,Clinic] -1 x[r,Instructor] -52 zV[r,rt,Clinic,Instructor] >= -42
 axisVdifRT[r,Location,Therapist]: +1 x[r,Location] -1 x[r,Therapist] -52 zV[r,rt,Location,Therapist]
  >= -42
 axisVdifRT[r,Location,Instructor]: +1 x[r,Location] -1 x[r,Instructor] -52 zV[r,rt,Location,Instructor]
  >= -42
 axisVdifRT[r,Therapist,Instructor]: +1 x[r,Therapist] -1 x[r,Instructor] -52 zV[r,rt,Therapist,Instructor]
  >= -42
 axisVdifRT[c,Client,CertificateTemplate]: +1 x[c,Client] -1 x[c,CertificateTemplate] -54 zV[c,rt,Client,CertificateTemplate]
  >= -42
 axisVdifRT[c,Client,Course]: +1 x[c,Client] -1 x[c,Course] -52 zV[c,rt,Client,Course] >= -42
 axisVdifRT[c,Client,Role]: +1 x[c,Client] -1 x[c,Role] -52 zV[c,rt,Client,Role] >= -42
 axisVdifRT[c,Client,User]: +1 x[c,Client] -1 x[c,User] -53 zV[c,rt,Client,User] >= -42
 axisVdifRT[c,Client,CourseSession]: +1 x[c,Client] -1 x[c,CourseSession] -56 zV[c,rt,Client,CourseSession]
  >= -42
 axisVdifRT[c,Client,Clinic]: +1 x[c,Client] -1 x[c,Clinic] -53 zV[c,rt,Client,Clinic] >= -42
 axisVdifRT[c,Client,Location]: +1 x[c,Client] -1 x[c,Location] -53 zV[c,rt,Client,Location] >= -42
 axisVdifRT[c,Client,Therapist]: +1 x[c,Client] -1 x[c,Therapist] -53 zV[c,rt,Client,Therapist] >= -42
 axisVdifRT[c,Client,Instructor]: +1 x[c,Client] -1 x[c,Instructor] -54 zV[c,rt,Client,Instructor] >= -42
 axisVdifRT[c,CertificateTemplate,Course]: +1 x[c,CertificateTemplate] -1 x[c,Course] -52 zV[c,rt,CertificateTemplate,Course]
  >= -42
 axisVdifRT[c,CertificateTemplate,Role]: +1 x[c,CertificateTemplate] -1 x[c,Role] -52 zV[c,rt,CertificateTemplate,Role]
  >= -42
 axisVdifRT[c,CertificateTemplate,User]: +1 x[c,CertificateTemplate] -1 x[c,User] -53 zV[c,rt,CertificateTemplate,User]
  >= -42
 axisVdifRT[c,CertificateTemplate,CourseSession]: +1 x[c,CertificateTemplate] -1 x[c,CourseSession] -56 zV[c,rt,CertificateTemplate,CourseSession]
  >= -42
 axisVdifRT[c,CertificateTemplate,Clinic]: +1 x[c,CertificateTemplate] -1 x[c,Clinic] -53 zV[c,rt,CertificateTemplate,Clinic]
  >= -42
 axisVdifRT[c,CertificateTemplate,Location]: +1 x[c,CertificateTemplate] -1 x[c,Location] -53 zV[c,rt,CertificateTemplate,Location]
  >= -42
 axisVdifRT[c,CertificateTemplate,Therapist]: +1 x[c,CertificateTemplate] -1 x[c,Therapist] -53 zV[c,rt,CertificateTemplate,Therapist]
  >= -42
 axisVdifRT[c,CertificateTemplate,Instructor]: +1 x[c,CertificateTemplate] -1 x[c,Instructor] -54 zV[c,rt,CertificateTemplate,Instructor]
  >= -42
 axisVdifRT[c,Course,Role]: +1 x[c,Course] -1 x[c,Role] -52 zV[c,rt,Course,Role] >= -42
 axisVdifRT[c,Course,User]: +1 x[c,Course] -1 x[c,User] -53 zV[c,rt,Course,User] >= -42
 axisVdifRT[c,Course,CourseSession]: +1 x[c,Course] -1 x[c,CourseSession] -56 zV[c,rt,Course,CourseSession]
  >= -42
 axisVdifRT[c,Course,Clinic]: +1 x[c,Course] -1 x[c,Clinic] -53 zV[c,rt,Course,Clinic] >= -42
 axisVdifRT[c,Course,Location]: +1 x[c,Course] -1 x[c,Location] -53 zV[c,rt,Course,Location] >= -42
 axisVdifRT[c,Course,Therapist]: +1 x[c,Course] -1 x[c,Therapist] -53 zV[c,rt,Course,Therapist] >= -42
 axisVdifRT[c,Course,Instructor]: +1 x[c,Course] -1 x[c,Instructor] -54 zV[c,rt,Course,Instructor] >= -42
 axisVdifRT[c,Role,User]: +1 x[c,Role] -1 x[c,User] -53 zV[c,rt,Role,User] >= -42
 axisVdifRT[c,Role,CourseSession]: +1 x[c,Role] -1 x[c,CourseSession] -56 zV[c,rt,Role,CourseSession]
  >= -42
 axisVdifRT[c,Role,Clinic]: +1 x[c,Role] -1 x[c,Clinic] -53 zV[c,rt,Role,Clinic] >= -42
 axisVdifRT[c,Role,Location]: +1 x[c,Role] -1 x[c,Location] -53 zV[c,rt,Role,Location] >= -42
 axisVdifRT[c,Role,Therapist]: +1 x[c,Role] -1 x[c,Therapist] -53 zV[c,rt,Role,Therapist] >= -42
 axisVdifRT[c,Role,Instructor]: +1 x[c,Role] -1 x[c,Instructor] -54 zV[c,rt,Role,Instructor] >= -42
 axisVdifRT[c,User,CourseSession]: +1 x[c,User] -1 x[c,CourseSession] -56 zV[c,rt,User,CourseSession]
  >= -42
 axisVdifRT[c,User,Clinic]: +1 x[c,User] -1 x[c,Clinic] -53 zV[c,rt,User,Clinic] >= -42
 axisVdifRT[c,User,Location]: +1 x[c,User] -1 x[c,Location] -53 zV[c,rt,User,Location] >= -42
 axisVdifRT[c,User,Therapist]: +1 x[c,User] -1 x[c,Therapist] -53 zV[c,rt,User,Therapist] >= -42
 axisVdifRT[c,User,Instructor]: +1 x[c,User] -1 x[c,Instructor] -54 zV[c,rt,User,Instructor] >= -42
 axisVdifRT[c,CourseSession,Clinic]: +1 x[c,CourseSession] -1 x[c,Clinic] -53 zV[c,rt,CourseSession,Clinic]
  >= -42
 axisVdifRT[c,CourseSession,Location]: +1 x[c,CourseSession] -1 x[c,Location] -53 zV[c,rt,CourseSession,Location]
  >= -42
 axisVdifRT[c,CourseSession,Therapist]: +1 x[c,CourseSession] -1 x[c,Therapist] -53 zV[c,rt,CourseSession,Therapist]
  >= -42
 axisVdifRT[c,CourseSession,Instructor]: +1 x[c,CourseSession] -1 x[c,Instructor] -54 zV[c,rt,CourseSession,Instructor]
  >= -42
 axisVdifRT[c,Clinic,Location]: +1 x[c,Clinic] -1 x[c,Location] -53 zV[c,rt,Clinic,Location] >= -42
 axisVdifRT[c,Clinic,Therapist]: +1 x[c,Clinic] -1 x[c,Therapist] -53 zV[c,rt,Clinic,Therapist] >= -42
 axisVdifRT[c,Clinic,Instructor]: +1 x[c,Clinic] -1 x[c,Instructor] -54 zV[c,rt,Clinic,Instructor] >= -42
 axisVdifRT[c,Location,Therapist]: +1 x[c,Location] -1 x[c,Therapist] -53 zV[c,rt,Location,Therapist]
  >= -42
 axisVdifRT[c,Location,Instructor]: +1 x[c,Location] -1 x[c,Instructor] -54 zV[c,rt,Location,Instructor]
  >= -42
 axisVdifRT[c,Therapist,Instructor]: +1 x[c,Therapist] -1 x[c,Instructor] -54 zV[c,rt,Therapist,Instructor]
  >= -42
 compDistAxis1[r,Client,CertificateTemplate]: -1 x[r,Client] +1 x[r,CertificateTemplate] +1 dist[r,Client,CertificateTemplate]
  >= +1
 compDistAxis1[r,Client,CourseSession]: -1 x[r,Client] +1 x[r,CourseSession] +1 dist[r,Client,CourseSession]
  >= +0
 compDistAxis1[r,Client,Instructor]: -1 x[r,Client] +1 x[r,Instructor] +1 dist[r,Client,Instructor] >= -0.5
 compDistAxis1[r,Client,Location]: -1 x[r,Client] +1 x[r,Location] +1 dist[r,Client,Location] >= +0
 compDistAxis1[r,Client,User]: -1 x[r,Client] +1 x[r,User] +1 dist[r,Client,User] >= +1
 compDistAxis1[r,CertificateTemplate,CourseSession]: -1 x[r,CertificateTemplate] +1 x[r,CourseSession]
  +1 dist[r,CertificateTemplate,CourseSession] >= -1
 compDistAxis1[r,Course,CourseSession]: -1 x[r,Course] +1 x[r,CourseSession] +1 dist[r,Course,CourseSession]
  >= -1.5
 compDistAxis1[r,CourseSession,Instructor]: -1 x[r,CourseSession] +1 x[r,Instructor] +1 dist[r,CourseSession,Instructor]
  >= -0.5
 compDistAxis1[r,Instructor,Location]: +1 x[r,Location] -1 x[r,Instructor] +1 dist[r,Instructor,Location]
  >= +0.5
 compDistAxis1[r,Clinic,Location]: -1 x[r,Clinic] +1 x[r,Location] +1 dist[r,Clinic,Location] >= -0.5
 compDistAxis1[r,Clinic,Therapist]: -1 x[r,Clinic] +1 x[r,Therapist] +1 dist[r,Clinic,Therapist] >= -1
 compDistAxis1[r,Location,Therapist]: -1 x[r,Location] +1 x[r,Therapist] +1 dist[r,Location,Therapist]
  >= -0.5
 compDistAxis1[r,Location,CourseSession]: +1 x[r,CourseSession] -1 x[r,Location] +1 dist[r,Location,CourseSession]
  >= +0
 compDistAxis1[r,Therapist,CourseSession]: +1 x[r,CourseSession] -1 x[r,Therapist] +1 dist[r,Therapist,CourseSession]
  >= +0.5
 compDistAxis1[r,Role,User]: -1 x[r,Role] +1 x[r,User] +1 dist[r,Role,User] >= -0.5
 compDistAxis1[r,User,Instructor]: -1 x[r,User] +1 x[r,Instructor] +1 dist[r,User,Instructor] >= -1.5
 compDistAxis1[r,User,Therapist]: -1 x[r,User] +1 x[r,Therapist] +1 dist[r,User,Therapist] >= -1.5
 compDistAxis1[c,Client,CertificateTemplate]: -1 x[c,Client] +1 x[c,CertificateTemplate] +1 dist[c,Client,CertificateTemplate]
  >= -0.5
 compDistAxis1[c,Client,CourseSession]: -1 x[c,Client] +1 x[c,CourseSession] +1 dist[c,Client,CourseSession]
  >= -1.5
 compDistAxis1[c,Client,Instructor]: -1 x[c,Client] +1 x[c,Instructor] +1 dist[c,Client,Instructor] >= -0.5
 compDistAxis1[c,Client,Location]: -1 x[c,Client] +1 x[c,Location] +1 dist[c,Client,Location] >= +0
 compDistAxis1[c,Client,User]: -1 x[c,Client] +1 x[c,User] +1 dist[c,Client,User] >= +0
 compDistAxis1[c,CertificateTemplate,CourseSession]: -1 x[c,CertificateTemplate] +1 x[c,CourseSession]
  +1 dist[c,CertificateTemplate,CourseSession] >= -1
 compDistAxis1[c,Course,CourseSession]: -1 x[c,Course] +1 x[c,CourseSession] +1 dist[c,Course,CourseSession]
  >= -2
 compDistAxis1[c,CourseSession,Instructor]: -1 x[c,CourseSession] +1 x[c,Instructor] +1 dist[c,CourseSession,Instructor]
  >= +1
 compDistAxis1[c,Instructor,Location]: +1 x[c,Location] -1 x[c,Instructor] +1 dist[c,Instructor,Location]
  >= +0.5
 compDistAxis1[c,Clinic,Location]: -1 x[c,Clinic] +1 x[c,Location] +1 dist[c,Clinic,Location] >= +0
 compDistAxis1[c,Clinic,Therapist]: -1 x[c,Clinic] +1 x[c,Therapist] +1 dist[c,Clinic,Therapist] >= +0
 compDistAxis1[c,Location,Therapist]: -1 x[c,Location] +1 x[c,Therapist] +1 dist[c,Location,Therapist]
  >= +0
 compDistAxis1[c,Location,CourseSession]: +1 x[c,CourseSession] -1 x[c,Location] +1 dist[c,Location,CourseSession]
  >= -1.5
 compDistAxis1[c,Therapist,CourseSession]: +1 x[c,CourseSession] -1 x[c,Therapist] +1 dist[c,Therapist,CourseSession]
  >= -1.5
 compDistAxis1[c,Role,User]: -1 x[c,Role] +1 x[c,User] +1 dist[c,Role,User] >= -0.5
 compDistAxis1[c,User,Instructor]: -1 x[c,User] +1 x[c,Instructor] +1 dist[c,User,Instructor] >= -0.5
 compDistAxis1[c,User,Therapist]: -1 x[c,User] +1 x[c,Therapist] +1 dist[c,User,Therapist] >= +0
 compDistAxis2[r,Client,CertificateTemplate]: +1 x[r,Client] -1 x[r,CertificateTemplate] +1 dist[r,Client,CertificateTemplate]
  >= -1
 compDistAxis2[r,Client,CourseSession]: +1 x[r,Client] -1 x[r,CourseSession] +1 dist[r,Client,CourseSession]
  >= +0
 compDistAxis2[r,Client,Instructor]: +1 x[r,Client] -1 x[r,Instructor] +1 dist[r,Client,Instructor] >= +0.5
 compDistAxis2[r,Client,Location]: +1 x[r,Client] -1 x[r,Location] +1 dist[r,Client,Location] >= +0
 compDistAxis2[r,Client,User]: +1 x[r,Client] -1 x[r,User] +1 dist[r,Client,User] >= -1
 compDistAxis2[r,CertificateTemplate,CourseSession]: +1 x[r,CertificateTemplate] -1 x[r,CourseSession]
  +1 dist[r,CertificateTemplate,CourseSession] >= +1
 compDistAxis2[r,Course,CourseSession]: +1 x[r,Course] -1 x[r,CourseSession] +1 dist[r,Course,CourseSession]
  >= +1.5
 compDistAxis2[r,CourseSession,Instructor]: +1 x[r,CourseSession] -1 x[r,Instructor] +1 dist[r,CourseSession,Instructor]
  >= +0.5
 compDistAxis2[r,Instructor,Location]: -1 x[r,Location] +1 x[r,Instructor] +1 dist[r,Instructor,Location]
  >= -0.5
 compDistAxis2[r,Clinic,Location]: +1 x[r,Clinic] -1 x[r,Location] +1 dist[r,Clinic,Location] >= +0.5
 compDistAxis2[r,Clinic,Therapist]: +1 x[r,Clinic] -1 x[r,Therapist] +1 dist[r,Clinic,Therapist] >= +1
 compDistAxis2[r,Location,Therapist]: +1 x[r,Location] -1 x[r,Therapist] +1 dist[r,Location,Therapist]
  >= +0.5
 compDistAxis2[r,Location,CourseSession]: -1 x[r,CourseSession] +1 x[r,Location] +1 dist[r,Location,CourseSession]
  >= +0
 compDistAxis2[r,Therapist,CourseSession]: -1 x[r,CourseSession] +1 x[r,Therapist] +1 dist[r,Therapist,CourseSession]
  >= -0.5
 compDistAxis2[r,Role,User]: +1 x[r,Role] -1 x[r,User] +1 dist[r,Role,User] >= +0.5
 compDistAxis2[r,User,Instructor]: +1 x[r,User] -1 x[r,Instructor] +1 dist[r,User,Instructor] >= +1.5
 compDistAxis2[r,User,Therapist]: +1 x[r,User] -1 x[r,Therapist] +1 dist[r,User,Therapist] >= +1.5
 compDistAxis2[c,Client,CertificateTemplate]: +1 x[c,Client] -1 x[c,CertificateTemplate] +1 dist[c,Client,CertificateTemplate]
  >= +0.5
 compDistAxis2[c,Client,CourseSession]: +1 x[c,Client] -1 x[c,CourseSession] +1 dist[c,Client,CourseSession]
  >= +1.5
 compDistAxis2[c,Client,Instructor]: +1 x[c,Client] -1 x[c,Instructor] +1 dist[c,Client,Instructor] >= +0.5
 compDistAxis2[c,Client,Location]: +1 x[c,Client] -1 x[c,Location] +1 dist[c,Client,Location] >= +0
 compDistAxis2[c,Client,User]: +1 x[c,Client] -1 x[c,User] +1 dist[c,Client,User] >= +0
 compDistAxis2[c,CertificateTemplate,CourseSession]: +1 x[c,CertificateTemplate] -1 x[c,CourseSession]
  +1 dist[c,CertificateTemplate,CourseSession] >= +1
 compDistAxis2[c,Course,CourseSession]: +1 x[c,Course] -1 x[c,CourseSession] +1 dist[c,Course,CourseSession]
  >= +2
 compDistAxis2[c,CourseSession,Instructor]: +1 x[c,CourseSession] -1 x[c,Instructor] +1 dist[c,CourseSession,Instructor]
  >= -1
 compDistAxis2[c,Instructor,Location]: -1 x[c,Location] +1 x[c,Instructor] +1 dist[c,Instructor,Location]
  >= -0.5
 compDistAxis2[c,Clinic,Location]: +1 x[c,Clinic] -1 x[c,Location] +1 dist[c,Clinic,Location] >= +0
 compDistAxis2[c,Clinic,Therapist]: +1 x[c,Clinic] -1 x[c,Therapist] +1 dist[c,Clinic,Therapist] >= +0
 compDistAxis2[c,Location,Therapist]: +1 x[c,Location] -1 x[c,Therapist] +1 dist[c,Location,Therapist]
  >= +0
 compDistAxis2[c,Location,CourseSession]: -1 x[c,CourseSession] +1 x[c,Location] +1 dist[c,Location,CourseSession]
  >= +1.5
 compDistAxis2[c,Therapist,CourseSession]: -1 x[c,CourseSession] +1 x[c,Therapist] +1 dist[c,Therapist,CourseSession]
  >= +1.5
 compDistAxis2[c,Role,User]: +1 x[c,Role] -1 x[c,User] +1 dist[c,Role,User] >= +0.5
 compDistAxis2[c,User,Instructor]: +1 x[c,User] -1 x[c,Instructor] +1 dist[c,User,Instructor] >= +0.5
 compDistAxis2[c,User,Therapist]: +1 x[c,User] -1 x[c,Therapist] +1 dist[c,User,Therapist] >= +0
 DistSumLB[Client,CertificateTemplate]: +1 dist[r,Client,CertificateTemplate] +1 dist[c,Client,CertificateTemplate]
  >= +8
 DistSumLB[Client,CourseSession]: +1 dist[r,Client,CourseSession] +1 dist[c,Client,CourseSession] >= +9
 DistSumLB[Client,Instructor]: +1 dist[r,Client,Instructor] +1 dist[c,Client,Instructor] >= +9.5
 DistSumLB[Client,Location]: +1 dist[r,Client,Location] +1 dist[c,Client,Location] >= +9
 DistSumLB[Client,User]: +1 dist[r,Client,User] +1 dist[c,Client,User] >= +8
 DistSumLB[CertificateTemplate,CourseSession]: +1 dist[r,CertificateTemplate,CourseSession] +1 dist[c,CertificateTemplate,CourseSession]
  >= +8
 DistSumLB[Course,CourseSession]: +1 dist[r,Course,CourseSession] +1 dist[c,Course,CourseSession] >= +7.5
 DistSumLB[CourseSession,Instructor]: +1 dist[r,CourseSession,Instructor] +1 dist[c,CourseSession,Instructor]
  >= +9.5
 DistSumLB[Instructor,Location]: +1 dist[r,Instructor,Location] +1 dist[c,Instructor,Location] >= +9.5
 DistSumLB[Clinic,Location]: +1 dist[r,Clinic,Location] +1 dist[c,Clinic,Location] >= +8.5
 DistSumLB[Clinic,Therapist]: +1 dist[r,Clinic,Therapist] +1 dist[c,Clinic,Therapist] >= +9
 DistSumLB[Location,Therapist]: +1 dist[r,Location,Therapist] +1 dist[c,Location,Therapist] >= +9.5
 DistSumLB[Location,CourseSession]: +1 dist[r,Location,CourseSession] +1 dist[c,Location,CourseSession]
  >= +9
 DistSumLB[Therapist,CourseSession]: +1 dist[r,Therapist,CourseSession] +1 dist[c,Therapist,CourseSession]
  >= +9.5
 DistSumLB[Role,User]: +1 dist[r,Role,User] +1 dist[c,Role,User] >= +6.5
 DistSumLB[User,Instructor]: +1 dist[r,User,Instructor] +1 dist[c,User,Instructor] >= +8.5
 DistSumLB[User,Therapist]: +1 dist[r,User,Therapist] +1 dist[c,User,Therapist] >= +8.5
 CenterDistSumLB: +1 center_dist[r,Client] +1 center_dist[r,CertificateTemplate] +1 center_dist[r,Course]
  +1 center_dist[r,Role] +1 center_dist[r,User] +1 center_dist[r,CourseSession] +1 center_dist[r,Clinic]
  +1 center_dist[r,Location] +1 center_dist[r,Therapist] +1 center_dist[r,Instructor] +1 center_dist[c,Client]
  +1 center_dist[c,CertificateTemplate] +1 center_dist[c,Course] +1 center_dist[c,Role] +1 center_dist[c,User]
  +1 center_dist[c,CourseSession] +1 center_dist[c,Clinic] +1 center_dist[c,Location] +1 center_dist[c,Therapist]
  +1 center_dist[c,Instructor] >= +30
 DistAxisLB1[r,Client,CertificateTemplate]: +1 dist[r,Client,CertificateTemplate] +8 zV[c,rt,Client,CertificateTemplate]
  +8 zV[c,lb,Client,CertificateTemplate] >= +8
 DistAxisLB1[r,Client,CourseSession]: +1 dist[r,Client,CourseSession] +9 zV[c,rt,Client,CourseSession]
  +9 zV[c,lb,Client,CourseSession] >= +9
 DistAxisLB1[r,Client,Instructor]: +1 dist[r,Client,Instructor] +9.5 zV[c,rt,Client,Instructor] +9.5 zV[c,lb,Client,Instructor]
  >= +9.5
 DistAxisLB1[r,Client,Location]: +1 dist[r,Client,Location] +9 zV[c,rt,Client,Location] +9 zV[c,lb,Client,Location]
  >= +9
 DistAxisLB1[r,Client,User]: +1 dist[r,Client,User] +8 zV[c,rt,Client,User] +8 zV[c,lb,Client,User] >= +8
 DistAxisLB1[r,CertificateTemplate,CourseSession]: +1 dist[r,CertificateTemplate,CourseSession] +8 zV[c,rt,CertificateTemplate,CourseSession]
  +8 zV[c,lb,CertificateTemplate,CourseSession] >= +8
 DistAxisLB1[r,Course,CourseSession]: +1 dist[r,Course,CourseSession] +7.5 zV[c,rt,Course,CourseSession]
  +7.5 zV[c,lb,Course,CourseSession] >= +7.5
 DistAxisLB1[r,CourseSession,Instructor]: +1 dist[r,CourseSession,Instructor] +9.5 zV[c,rt,CourseSession,Instructor]
  +9.5 zV[c,lb,CourseSession,Instructor] >= +9.5
 DistAxisLB1[r,Instructor,Location]: +1 dist[r,Instructor,Location] +9.5 zV[c,rt,Location,Instructor]
  +9.5 zV[c,lb,Location,Instructor] >= +9.5
 DistAxisLB1[r,Clinic,Location]: +1 dist[r,Clinic,Location] +8.5 zV[c,rt,Clinic,Location] +8.5 zV[c,lb,Clinic,Location]
  >= +8.5
 DistAxisLB1[r,Clinic,Therapist]: +1 dist[r,Clinic,Therapist] +9 zV[c,rt,Clinic,Therapist] +9 zV[c,lb,Clinic,Therapist]
  >= +9
 DistAxisLB1[r,Location,Therapist]: +1 dist[r,Location,Therapist] +9.5 zV[c,rt,Location,Therapist] +9.5 zV[c,lb,Location,Therapist]
  >= +9.5
 DistAxisLB1[r,Location,CourseSession]: +1 dist[r,Location,CourseSession] +9 zV[c,rt,CourseSession,Location]
  +9 zV[c,lb,CourseSession,Location] >= +9
 DistAxisLB1[r,Therapist,CourseSession]: +1 dist[r,Therapist,CourseSession] +9.5 zV[c,rt,CourseSession,Therapist]
  +9.5 zV[c,lb,CourseSession,Therapist] >= +9.5
 DistAxisLB1[r,Role,User]: +1 dist[r,Role,User] +6.5 zV[c,rt,Role,User] +6.5 zV[c,lb,Role,User] >= +6.5
 DistAxisLB1[r,User,Instructor]: +1 dist[r,User,Instructor] +8.5 zV[c,rt,User,Instructor] +8.5 zV[c,lb,User,Instructor]
  >= +8.5
 DistAxisLB1[r,User,Therapist]: +1 dist[r,User,Therapist] +8.5 zV[c,rt,User,Therapist] +8.5 zV[c,lb,User,Therapist]
  >= +8.5
 DistAxisLB1[c,Client,CertificateTemplate]: +1 dist[c,Client,CertificateTemplate] +11.5 zV[r,rt,Client,CertificateTemplate]
  +11.5 zV[r,lb,Client,CertificateTemplate] >= +11.5
 DistAxisLB1[c,Client,CourseSession]: +1 dist[c,Client,CourseSession] +12.5 zV[r,rt,Client,CourseSession]
  +12.5 zV[r,lb,Client,CourseSession] >= +12.5
 DistAxisLB1[c,Client,Instructor]: +1 dist[c,Client,Instructor] +11.5 zV[r,rt,Client,Instructor] +11.5 zV[r,lb,Client,Instructor]
  >= +11.5
 DistAxisLB1[c,Client,Location]: +1 dist[c,Client,Location] +11 zV[r,rt,Client,Location] +11 zV[r,lb,Client,Location]
  >= +11
 DistAxisLB1[c,Client,User]: +1 dist[c,Client,User] +11 zV[r,rt,Client,User] +11 zV[r,lb,Client,User]
  >= +11
 DistAxisLB1[c,CertificateTemplate,CourseSession]: +1 dist[c,CertificateTemplate,CourseSession] +13 zV[r,rt,CertificateTemplate,CourseSession]
  +13 zV[r,lb,CertificateTemplate,CourseSession] >= +13
 DistAxisLB1[c,Course,CourseSession]: +1 dist[c,Course,CourseSession] +12 zV[r,rt,Course,CourseSession]
  +12 zV[r,lb,Course,CourseSession] >= +12
 DistAxisLB1[c,CourseSession,Instructor]: +1 dist[c,CourseSession,Instructor] +13 zV[r,rt,CourseSession,Instructor]
  +13 zV[r,lb,CourseSession,Instructor] >= +13
 DistAxisLB1[c,Instructor,Location]: +1 dist[c,Instructor,Location] +11.5 zV[r,rt,Location,Instructor]
  +11.5 zV[r,lb,Location,Instructor] >= +11.5
 DistAxisLB1[c,Clinic,Location]: +1 dist[c,Clinic,Location] +11 zV[r,rt,Clinic,Location] +11 zV[r,lb,Clinic,Location]
  >= +11
 DistAxisLB1[c,Clinic,Therapist]: +1 dist[c,Clinic,Therapist] +11 zV[r,rt,Clinic,Therapist] +11 zV[r,lb,Clinic,Therapist]
  >= +11
 DistAxisLB1[c,Location,Therapist]: +1 dist[c,Location,Therapist] +11 zV[r,rt,Location,Therapist] +11 zV[r,lb,Location,Therapist]
  >= +11
 DistAxisLB1[c,Location,CourseSession]: +1 dist[c,Location,CourseSession] +12.5 zV[r,rt,CourseSession,Location]
  +12.5 zV[r,lb,CourseSession,Location] >= +12.5
 DistAxisLB1[c,Therapist,CourseSession]: +1 dist[c,Therapist,CourseSession] +12.5 zV[r,rt,CourseSession,Therapist]
  +12.5 zV[r,lb,CourseSession,Therapist] >= +12.5
 DistAxisLB1[c,Role,User]: +1 dist[c,Role,User] +10.5 zV[r,rt,Role,User] +10.5 zV[r,lb,Role,User] >= +10.5
 DistAxisLB1[c,User,Instructor]: +1 dist[c,User,Instructor] +11.5 zV[r,rt,User,Instructor] +11.5 zV[r,lb,User,Instructor]
  >= +11.5
 DistAxisLB1[c,User,Therapist]: +1 dist[c,User,Therapist] +11 zV[r,rt,User,Therapist] +11 zV[r,lb,User,Therapist]
  >= +11
 triangle1[Client,CertificateTemplate,Course,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Course]
  +1 zV[r,rt,CertificateTemplate,Course] <= +2
 triangle1[Client,CertificateTemplate,Course,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Course]
  +1 zV[c,rt,CertificateTemplate,Course] <= +2
 triangle1[Client,CertificateTemplate,Role,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Role]
  +1 zV[r,rt,CertificateTemplate,Role] <= +2
 triangle1[Client,CertificateTemplate,Role,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Role]
  +1 zV[c,rt,CertificateTemplate,Role] <= +2
 triangle1[Client,CertificateTemplate,User,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,User]
  +1 zV[r,rt,CertificateTemplate,User] <= +2
 triangle1[Client,CertificateTemplate,User,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,User]
  +1 zV[c,rt,CertificateTemplate,User] <= +2
 triangle1[Client,CertificateTemplate,CourseSession,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,CourseSession]
  +1 zV[r,rt,CertificateTemplate,CourseSession] <= +2
 triangle1[Client,CertificateTemplate,CourseSession,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,CourseSession]
  +1 zV[c,rt,CertificateTemplate,CourseSession] <= +2
 triangle1[Client,CertificateTemplate,Clinic,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Clinic]
  +1 zV[r,rt,CertificateTemplate,Clinic] <= +2
 triangle1[Client,CertificateTemplate,Clinic,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Clinic]
  +1 zV[c,rt,CertificateTemplate,Clinic] <= +2
 triangle1[Client,CertificateTemplate,Location,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Location]
  +1 zV[r,rt,CertificateTemplate,Location] <= +2
 triangle1[Client,CertificateTemplate,Location,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Location]
  +1 zV[c,rt,CertificateTemplate,Location] <= +2
 triangle1[Client,CertificateTemplate,Therapist,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Therapist]
  +1 zV[r,rt,CertificateTemplate,Therapist] <= +2
 triangle1[Client,CertificateTemplate,Therapist,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Therapist]
  +1 zV[c,rt,CertificateTemplate,Therapist] <= +2
 triangle1[Client,CertificateTemplate,Instructor,r]: +1 zV[r,rt,Client,CertificateTemplate] +1 zV[r,lb,Client,Instructor]
  +1 zV[r,rt,CertificateTemplate,Instructor] <= +2
 triangle1[Client,CertificateTemplate,Instructor,c]: +1 zV[c,rt,Client,CertificateTemplate] +1 zV[c,lb,Client,Instructor]
  +1 zV[c,rt,CertificateTemplate,Instructor] <= +2
 triangle1[Client,Course,Role,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,Role] +1 zV[r,rt,Course,Role]
  <= +2
 triangle1[Client,Course,Role,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,Role] +1 zV[c,rt,Course,Role]
  <= +2
 triangle1[Client,Course,User,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,User] +1 zV[r,rt,Course,User]
  <= +2
 triangle1[Client,Course,User,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,User] +1 zV[c,rt,Course,User]
  <= +2
 triangle1[Client,Course,CourseSession,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,CourseSession]
  +1 zV[r,rt,Course,CourseSession] <= +2
 triangle1[Client,Course,CourseSession,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,CourseSession]
  +1 zV[c,rt,Course,CourseSession] <= +2
 triangle1[Client,Course,Clinic,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,Course,Clinic]
  <= +2
 triangle1[Client,Course,Clinic,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,Course,Clinic]
  <= +2
 triangle1[Client,Course,Location,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,Location] +1 zV[r,rt,Course,Location]
  <= +2
 triangle1[Client,Course,Location,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,Location] +1 zV[c,rt,Course,Location]
  <= +2
 triangle1[Client,Course,Therapist,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,Course,Therapist]
  <= +2
 triangle1[Client,Course,Therapist,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,Course,Therapist]
  <= +2
 triangle1[Client,Course,Instructor,r]: +1 zV[r,rt,Client,Course] +1 zV[r,lb,Client,Instructor] +1 zV[r,rt,Course,Instructor]
  <= +2
 triangle1[Client,Course,Instructor,c]: +1 zV[c,rt,Client,Course] +1 zV[c,lb,Client,Instructor] +1 zV[c,rt,Course,Instructor]
  <= +2
 triangle1[Client,Role,User,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,User] +1 zV[r,rt,Role,User]
  <= +2
 triangle1[Client,Role,User,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,User] +1 zV[c,rt,Role,User]
  <= +2
 triangle1[Client,Role,CourseSession,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,Role,CourseSession]
  <= +2
 triangle1[Client,Role,CourseSession,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,Role,CourseSession]
  <= +2
 triangle1[Client,Role,Clinic,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,Role,Clinic]
  <= +2
 triangle1[Client,Role,Clinic,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,Role,Clinic]
  <= +2
 triangle1[Client,Role,Location,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,Location] +1 zV[r,rt,Role,Location]
  <= +2
 triangle1[Client,Role,Location,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,Location] +1 zV[c,rt,Role,Location]
  <= +2
 triangle1[Client,Role,Therapist,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,Role,Therapist]
  <= +2
 triangle1[Client,Role,Therapist,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,Role,Therapist]
  <= +2
 triangle1[Client,Role,Instructor,r]: +1 zV[r,rt,Client,Role] +1 zV[r,lb,Client,Instructor] +1 zV[r,rt,Role,Instructor]
  <= +2
 triangle1[Client,Role,Instructor,c]: +1 zV[c,rt,Client,Role] +1 zV[c,lb,Client,Instructor] +1 zV[c,rt,Role,Instructor]
  <= +2
 triangle1[Client,User,CourseSession,r]: +1 zV[r,rt,Client,User] +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,User,CourseSession]
  <= +2
 triangle1[Client,User,CourseSession,c]: +1 zV[c,rt,Client,User] +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,User,CourseSession]
  <= +2
 triangle1[Client,User,Clinic,r]: +1 zV[r,rt,Client,User] +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,User,Clinic]
  <= +2
 triangle1[Client,User,Clinic,c]: +1 zV[c,rt,Client,User] +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,User,Clinic]
  <= +2
 triangle1[Client,User,Location,r]: +1 zV[r,rt,Client,User] +1 zV[r,lb,Client,Location] +1 zV[r,rt,User,Location]
  <= +2
 triangle1[Client,User,Location,c]: +1 zV[c,rt,Client,User] +1 zV[c,lb,Client,Location] +1 zV[c,rt,User,Location]
  <= +2
 triangle1[Client,User,Therapist,r]: +1 zV[r,rt,Client,User] +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,User,Therapist]
  <= +2
 triangle1[Client,User,Therapist,c]: +1 zV[c,rt,Client,User] +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,User,Therapist]
  <= +2
 triangle1[Client,User,Instructor,r]: +1 zV[r,rt,Client,User] +1 zV[r,lb,Client,Instructor] +1 zV[r,rt,User,Instructor]
  <= +2
 triangle1[Client,User,Instructor,c]: +1 zV[c,rt,Client,User] +1 zV[c,lb,Client,Instructor] +1 zV[c,rt,User,Instructor]
  <= +2
 triangle1[Client,CourseSession,Clinic,r]: +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,Client,Clinic]
  +1 zV[r,rt,CourseSession,Clinic] <= +2
 triangle1[Client,CourseSession,Clinic,c]: +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,Client,Clinic]
  +1 zV[c,rt,CourseSession,Clinic] <= +2
 triangle1[Client,CourseSession,Location,r]: +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,Client,Location]
  +1 zV[r,rt,CourseSession,Location] <= +2
 triangle1[Client,CourseSession,Location,c]: +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,Client,Location]
  +1 zV[c,rt,CourseSession,Location] <= +2
 triangle1[Client,CourseSession,Therapist,r]: +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,Client,Therapist]
  +1 zV[r,rt,CourseSession,Therapist] <= +2
 triangle1[Client,CourseSession,Therapist,c]: +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,Client,Therapist]
  +1 zV[c,rt,CourseSession,Therapist] <= +2
 triangle1[Client,CourseSession,Instructor,r]: +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,Client,Instructor]
  +1 zV[r,rt,CourseSession,Instructor] <= +2
 triangle1[Client,CourseSession,Instructor,c]: +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,Client,Instructor]
  +1 zV[c,rt,CourseSession,Instructor] <= +2
 triangle1[Client,Clinic,Location,r]: +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,Client,Location] +1 zV[r,rt,Clinic,Location]
  <= +2
 triangle1[Client,Clinic,Location,c]: +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,Client,Location] +1 zV[c,rt,Clinic,Location]
  <= +2
 triangle1[Client,Clinic,Therapist,r]: +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,Clinic,Therapist]
  <= +2
 triangle1[Client,Clinic,Therapist,c]: +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,Clinic,Therapist]
  <= +2
 triangle1[Client,Clinic,Instructor,r]: +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,Client,Instructor] +1 zV[r,rt,Clinic,Instructor]
  <= +2
 triangle1[Client,Clinic,Instructor,c]: +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,Client,Instructor] +1 zV[c,rt,Clinic,Instructor]
  <= +2
 triangle1[Client,Location,Therapist,r]: +1 zV[r,rt,Client,Location] +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,Location,Therapist]
  <= +2
 triangle1[Client,Location,Therapist,c]: +1 zV[c,rt,Client,Location] +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,Location,Therapist]
  <= +2
 triangle1[Client,Location,Instructor,r]: +1 zV[r,rt,Client,Location] +1 zV[r,lb,Client,Instructor] +1 zV[r,rt,Location,Instructor]
  <= +2
 triangle1[Client,Location,Instructor,c]: +1 zV[c,rt,Client,Location] +1 zV[c,lb,Client,Instructor] +1 zV[c,rt,Location,Instructor]
  <= +2
 triangle1[Client,Therapist,Instructor,r]: +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,Client,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[Client,Therapist,Instructor,c]: +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,Client,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle1[CertificateTemplate,Course,Role,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,Role]
  +1 zV[r,rt,Course,Role] <= +2
 triangle1[CertificateTemplate,Course,Role,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Role]
  +1 zV[c,rt,Course,Role] <= +2
 triangle1[CertificateTemplate,Course,User,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,User]
  +1 zV[r,rt,Course,User] <= +2
 triangle1[CertificateTemplate,Course,User,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,User]
  +1 zV[c,rt,Course,User] <= +2
 triangle1[CertificateTemplate,Course,CourseSession,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,Course,CourseSession] <= +2
 triangle1[CertificateTemplate,Course,CourseSession,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,Course,CourseSession] <= +2
 triangle1[CertificateTemplate,Course,Clinic,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,Clinic]
  +1 zV[r,rt,Course,Clinic] <= +2
 triangle1[CertificateTemplate,Course,Clinic,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Clinic]
  +1 zV[c,rt,Course,Clinic] <= +2
 triangle1[CertificateTemplate,Course,Location,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,Location]
  +1 zV[r,rt,Course,Location] <= +2
 triangle1[CertificateTemplate,Course,Location,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Location]
  +1 zV[c,rt,Course,Location] <= +2
 triangle1[CertificateTemplate,Course,Therapist,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,Therapist]
  +1 zV[r,rt,Course,Therapist] <= +2
 triangle1[CertificateTemplate,Course,Therapist,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Therapist]
  +1 zV[c,rt,Course,Therapist] <= +2
 triangle1[CertificateTemplate,Course,Instructor,r]: +1 zV[r,rt,CertificateTemplate,Course] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,Course,Instructor] <= +2
 triangle1[CertificateTemplate,Course,Instructor,c]: +1 zV[c,rt,CertificateTemplate,Course] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,Course,Instructor] <= +2
 triangle1[CertificateTemplate,Role,User,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,User]
  +1 zV[r,rt,Role,User] <= +2
 triangle1[CertificateTemplate,Role,User,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,User]
  +1 zV[c,rt,Role,User] <= +2
 triangle1[CertificateTemplate,Role,CourseSession,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,Role,CourseSession] <= +2
 triangle1[CertificateTemplate,Role,CourseSession,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,Role,CourseSession] <= +2
 triangle1[CertificateTemplate,Role,Clinic,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,Clinic]
  +1 zV[r,rt,Role,Clinic] <= +2
 triangle1[CertificateTemplate,Role,Clinic,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,Clinic]
  +1 zV[c,rt,Role,Clinic] <= +2
 triangle1[CertificateTemplate,Role,Location,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,Location]
  +1 zV[r,rt,Role,Location] <= +2
 triangle1[CertificateTemplate,Role,Location,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,Location]
  +1 zV[c,rt,Role,Location] <= +2
 triangle1[CertificateTemplate,Role,Therapist,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,Therapist]
  +1 zV[r,rt,Role,Therapist] <= +2
 triangle1[CertificateTemplate,Role,Therapist,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,Therapist]
  +1 zV[c,rt,Role,Therapist] <= +2
 triangle1[CertificateTemplate,Role,Instructor,r]: +1 zV[r,rt,CertificateTemplate,Role] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,Role,Instructor] <= +2
 triangle1[CertificateTemplate,Role,Instructor,c]: +1 zV[c,rt,CertificateTemplate,Role] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,Role,Instructor] <= +2
 triangle1[CertificateTemplate,User,CourseSession,r]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,User,CourseSession] <= +2
 triangle1[CertificateTemplate,User,CourseSession,c]: +1 zV[c,rt,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,User,CourseSession] <= +2
 triangle1[CertificateTemplate,User,Clinic,r]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[r,lb,CertificateTemplate,Clinic]
  +1 zV[r,rt,User,Clinic] <= +2
 triangle1[CertificateTemplate,User,Clinic,c]: +1 zV[c,rt,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,Clinic]
  +1 zV[c,rt,User,Clinic] <= +2
 triangle1[CertificateTemplate,User,Location,r]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[r,lb,CertificateTemplate,Location]
  +1 zV[r,rt,User,Location] <= +2
 triangle1[CertificateTemplate,User,Location,c]: +1 zV[c,rt,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,Location]
  +1 zV[c,rt,User,Location] <= +2
 triangle1[CertificateTemplate,User,Therapist,r]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[r,lb,CertificateTemplate,Therapist]
  +1 zV[r,rt,User,Therapist] <= +2
 triangle1[CertificateTemplate,User,Therapist,c]: +1 zV[c,rt,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,Therapist]
  +1 zV[c,rt,User,Therapist] <= +2
 triangle1[CertificateTemplate,User,Instructor,r]: +1 zV[r,rt,CertificateTemplate,User] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,User,Instructor] <= +2
 triangle1[CertificateTemplate,User,Instructor,c]: +1 zV[c,rt,CertificateTemplate,User] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,User,Instructor] <= +2
 triangle1[CertificateTemplate,CourseSession,Clinic,r]: +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,CertificateTemplate,Clinic] +1 zV[r,rt,CourseSession,Clinic] <= +2
 triangle1[CertificateTemplate,CourseSession,Clinic,c]: +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,CertificateTemplate,Clinic] +1 zV[c,rt,CourseSession,Clinic] <= +2
 triangle1[CertificateTemplate,CourseSession,Location,r]: +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,CertificateTemplate,Location] +1 zV[r,rt,CourseSession,Location] <= +2
 triangle1[CertificateTemplate,CourseSession,Location,c]: +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,CertificateTemplate,Location] +1 zV[c,rt,CourseSession,Location] <= +2
 triangle1[CertificateTemplate,CourseSession,Therapist,r]: +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,CertificateTemplate,Therapist] +1 zV[r,rt,CourseSession,Therapist] <= +2
 triangle1[CertificateTemplate,CourseSession,Therapist,c]: +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,CertificateTemplate,Therapist] +1 zV[c,rt,CourseSession,Therapist] <= +2
 triangle1[CertificateTemplate,CourseSession,Instructor,r]: +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,CertificateTemplate,Instructor] +1 zV[r,rt,CourseSession,Instructor] <= +2
 triangle1[CertificateTemplate,CourseSession,Instructor,c]: +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,CertificateTemplate,Instructor] +1 zV[c,rt,CourseSession,Instructor] <= +2
 triangle1[CertificateTemplate,Clinic,Location,r]: +1 zV[r,rt,CertificateTemplate,Clinic] +1 zV[r,lb,CertificateTemplate,Location]
  +1 zV[r,rt,Clinic,Location] <= +2
 triangle1[CertificateTemplate,Clinic,Location,c]: +1 zV[c,rt,CertificateTemplate,Clinic] +1 zV[c,lb,CertificateTemplate,Location]
  +1 zV[c,rt,Clinic,Location] <= +2
 triangle1[CertificateTemplate,Clinic,Therapist,r]: +1 zV[r,rt,CertificateTemplate,Clinic] +1 zV[r,lb,CertificateTemplate,Therapist]
  +1 zV[r,rt,Clinic,Therapist] <= +2
 triangle1[CertificateTemplate,Clinic,Therapist,c]: +1 zV[c,rt,CertificateTemplate,Clinic] +1 zV[c,lb,CertificateTemplate,Therapist]
  +1 zV[c,rt,Clinic,Therapist] <= +2
 triangle1[CertificateTemplate,Clinic,Instructor,r]: +1 zV[r,rt,CertificateTemplate,Clinic] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,Clinic,Instructor] <= +2
 triangle1[CertificateTemplate,Clinic,Instructor,c]: +1 zV[c,rt,CertificateTemplate,Clinic] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,Clinic,Instructor] <= +2
 triangle1[CertificateTemplate,Location,Therapist,r]: +1 zV[r,rt,CertificateTemplate,Location] +1 zV[r,lb,CertificateTemplate,Therapist]
  +1 zV[r,rt,Location,Therapist] <= +2
 triangle1[CertificateTemplate,Location,Therapist,c]: +1 zV[c,rt,CertificateTemplate,Location] +1 zV[c,lb,CertificateTemplate,Therapist]
  +1 zV[c,rt,Location,Therapist] <= +2
 triangle1[CertificateTemplate,Location,Instructor,r]: +1 zV[r,rt,CertificateTemplate,Location] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,Location,Instructor] <= +2
 triangle1[CertificateTemplate,Location,Instructor,c]: +1 zV[c,rt,CertificateTemplate,Location] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,Location,Instructor] <= +2
 triangle1[CertificateTemplate,Therapist,Instructor,r]: +1 zV[r,rt,CertificateTemplate,Therapist] +1 zV[r,lb,CertificateTemplate,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[CertificateTemplate,Therapist,Instructor,c]: +1 zV[c,rt,CertificateTemplate,Therapist] +1 zV[c,lb,CertificateTemplate,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle1[Course,Role,User,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,User] +1 zV[r,rt,Role,User]
  <= +2
 triangle1[Course,Role,User,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,User] +1 zV[c,rt,Role,User]
  <= +2
 triangle1[Course,Role,CourseSession,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,Role,CourseSession]
  <= +2
 triangle1[Course,Role,CourseSession,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,Role,CourseSession]
  <= +2
 triangle1[Course,Role,Clinic,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,Clinic] +1 zV[r,rt,Role,Clinic]
  <= +2
 triangle1[Course,Role,Clinic,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,Clinic] +1 zV[c,rt,Role,Clinic]
  <= +2
 triangle1[Course,Role,Location,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,Location] +1 zV[r,rt,Role,Location]
  <= +2
 triangle1[Course,Role,Location,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,Location] +1 zV[c,rt,Role,Location]
  <= +2
 triangle1[Course,Role,Therapist,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,Therapist] +1 zV[r,rt,Role,Therapist]
  <= +2
 triangle1[Course,Role,Therapist,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,Therapist] +1 zV[c,rt,Role,Therapist]
  <= +2
 triangle1[Course,Role,Instructor,r]: +1 zV[r,rt,Course,Role] +1 zV[r,lb,Course,Instructor] +1 zV[r,rt,Role,Instructor]
  <= +2
 triangle1[Course,Role,Instructor,c]: +1 zV[c,rt,Course,Role] +1 zV[c,lb,Course,Instructor] +1 zV[c,rt,Role,Instructor]
  <= +2
 triangle1[Course,User,CourseSession,r]: +1 zV[r,rt,Course,User] +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,User,CourseSession]
  <= +2
 triangle1[Course,User,CourseSession,c]: +1 zV[c,rt,Course,User] +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,User,CourseSession]
  <= +2
 triangle1[Course,User,Clinic,r]: +1 zV[r,rt,Course,User] +1 zV[r,lb,Course,Clinic] +1 zV[r,rt,User,Clinic]
  <= +2
 triangle1[Course,User,Clinic,c]: +1 zV[c,rt,Course,User] +1 zV[c,lb,Course,Clinic] +1 zV[c,rt,User,Clinic]
  <= +2
 triangle1[Course,User,Location,r]: +1 zV[r,rt,Course,User] +1 zV[r,lb,Course,Location] +1 zV[r,rt,User,Location]
  <= +2
 triangle1[Course,User,Location,c]: +1 zV[c,rt,Course,User] +1 zV[c,lb,Course,Location] +1 zV[c,rt,User,Location]
  <= +2
 triangle1[Course,User,Therapist,r]: +1 zV[r,rt,Course,User] +1 zV[r,lb,Course,Therapist] +1 zV[r,rt,User,Therapist]
  <= +2
 triangle1[Course,User,Therapist,c]: +1 zV[c,rt,Course,User] +1 zV[c,lb,Course,Therapist] +1 zV[c,rt,User,Therapist]
  <= +2
 triangle1[Course,User,Instructor,r]: +1 zV[r,rt,Course,User] +1 zV[r,lb,Course,Instructor] +1 zV[r,rt,User,Instructor]
  <= +2
 triangle1[Course,User,Instructor,c]: +1 zV[c,rt,Course,User] +1 zV[c,lb,Course,Instructor] +1 zV[c,rt,User,Instructor]
  <= +2
 triangle1[Course,CourseSession,Clinic,r]: +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,Course,Clinic]
  +1 zV[r,rt,CourseSession,Clinic] <= +2
 triangle1[Course,CourseSession,Clinic,c]: +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,Course,Clinic]
  +1 zV[c,rt,CourseSession,Clinic] <= +2
 triangle1[Course,CourseSession,Location,r]: +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,Course,Location]
  +1 zV[r,rt,CourseSession,Location] <= +2
 triangle1[Course,CourseSession,Location,c]: +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,Course,Location]
  +1 zV[c,rt,CourseSession,Location] <= +2
 triangle1[Course,CourseSession,Therapist,r]: +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,Course,Therapist]
  +1 zV[r,rt,CourseSession,Therapist] <= +2
 triangle1[Course,CourseSession,Therapist,c]: +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,Course,Therapist]
  +1 zV[c,rt,CourseSession,Therapist] <= +2
 triangle1[Course,CourseSession,Instructor,r]: +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,Course,Instructor]
  +1 zV[r,rt,CourseSession,Instructor] <= +2
 triangle1[Course,CourseSession,Instructor,c]: +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,Course,Instructor]
  +1 zV[c,rt,CourseSession,Instructor] <= +2
 triangle1[Course,Clinic,Location,r]: +1 zV[r,rt,Course,Clinic] +1 zV[r,lb,Course,Location] +1 zV[r,rt,Clinic,Location]
  <= +2
 triangle1[Course,Clinic,Location,c]: +1 zV[c,rt,Course,Clinic] +1 zV[c,lb,Course,Location] +1 zV[c,rt,Clinic,Location]
  <= +2
 triangle1[Course,Clinic,Therapist,r]: +1 zV[r,rt,Course,Clinic] +1 zV[r,lb,Course,Therapist] +1 zV[r,rt,Clinic,Therapist]
  <= +2
 triangle1[Course,Clinic,Therapist,c]: +1 zV[c,rt,Course,Clinic] +1 zV[c,lb,Course,Therapist] +1 zV[c,rt,Clinic,Therapist]
  <= +2
 triangle1[Course,Clinic,Instructor,r]: +1 zV[r,rt,Course,Clinic] +1 zV[r,lb,Course,Instructor] +1 zV[r,rt,Clinic,Instructor]
  <= +2
 triangle1[Course,Clinic,Instructor,c]: +1 zV[c,rt,Course,Clinic] +1 zV[c,lb,Course,Instructor] +1 zV[c,rt,Clinic,Instructor]
  <= +2
 triangle1[Course,Location,Therapist,r]: +1 zV[r,rt,Course,Location] +1 zV[r,lb,Course,Therapist] +1 zV[r,rt,Location,Therapist]
  <= +2
 triangle1[Course,Location,Therapist,c]: +1 zV[c,rt,Course,Location] +1 zV[c,lb,Course,Therapist] +1 zV[c,rt,Location,Therapist]
  <= +2
 triangle1[Course,Location,Instructor,r]: +1 zV[r,rt,Course,Location] +1 zV[r,lb,Course,Instructor] +1 zV[r,rt,Location,Instructor]
  <= +2
 triangle1[Course,Location,Instructor,c]: +1 zV[c,rt,Course,Location] +1 zV[c,lb,Course,Instructor] +1 zV[c,rt,Location,Instructor]
  <= +2
 triangle1[Course,Therapist,Instructor,r]: +1 zV[r,rt,Course,Therapist] +1 zV[r,lb,Course,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[Course,Therapist,Instructor,c]: +1 zV[c,rt,Course,Therapist] +1 zV[c,lb,Course,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle1[Role,User,CourseSession,r]: +1 zV[r,rt,Role,User] +1 zV[r,lb,Role,CourseSession] +1 zV[r,rt,User,CourseSession]
  <= +2
 triangle1[Role,User,CourseSession,c]: +1 zV[c,rt,Role,User] +1 zV[c,lb,Role,CourseSession] +1 zV[c,rt,User,CourseSession]
  <= +2
 triangle1[Role,User,Clinic,r]: +1 zV[r,rt,Role,User] +1 zV[r,lb,Role,Clinic] +1 zV[r,rt,User,Clinic]
  <= +2
 triangle1[Role,User,Clinic,c]: +1 zV[c,rt,Role,User] +1 zV[c,lb,Role,Clinic] +1 zV[c,rt,User,Clinic]
  <= +2
 triangle1[Role,User,Location,r]: +1 zV[r,rt,Role,User] +1 zV[r,lb,Role,Location] +1 zV[r,rt,User,Location]
  <= +2
 triangle1[Role,User,Location,c]: +1 zV[c,rt,Role,User] +1 zV[c,lb,Role,Location] +1 zV[c,rt,User,Location]
  <= +2
 triangle1[Role,User,Therapist,r]: +1 zV[r,rt,Role,User] +1 zV[r,lb,Role,Therapist] +1 zV[r,rt,User,Therapist]
  <= +2
 triangle1[Role,User,Therapist,c]: +1 zV[c,rt,Role,User] +1 zV[c,lb,Role,Therapist] +1 zV[c,rt,User,Therapist]
  <= +2
 triangle1[Role,User,Instructor,r]: +1 zV[r,rt,Role,User] +1 zV[r,lb,Role,Instructor] +1 zV[r,rt,User,Instructor]
  <= +2
 triangle1[Role,User,Instructor,c]: +1 zV[c,rt,Role,User] +1 zV[c,lb,Role,Instructor] +1 zV[c,rt,User,Instructor]
  <= +2
 triangle1[Role,CourseSession,Clinic,r]: +1 zV[r,rt,Role,CourseSession] +1 zV[r,lb,Role,Clinic] +1 zV[r,rt,CourseSession,Clinic]
  <= +2
 triangle1[Role,CourseSession,Clinic,c]: +1 zV[c,rt,Role,CourseSession] +1 zV[c,lb,Role,Clinic] +1 zV[c,rt,CourseSession,Clinic]
  <= +2
 triangle1[Role,CourseSession,Location,r]: +1 zV[r,rt,Role,CourseSession] +1 zV[r,lb,Role,Location] +1 zV[r,rt,CourseSession,Location]
  <= +2
 triangle1[Role,CourseSession,Location,c]: +1 zV[c,rt,Role,CourseSession] +1 zV[c,lb,Role,Location] +1 zV[c,rt,CourseSession,Location]
  <= +2
 triangle1[Role,CourseSession,Therapist,r]: +1 zV[r,rt,Role,CourseSession] +1 zV[r,lb,Role,Therapist]
  +1 zV[r,rt,CourseSession,Therapist] <= +2
 triangle1[Role,CourseSession,Therapist,c]: +1 zV[c,rt,Role,CourseSession] +1 zV[c,lb,Role,Therapist]
  +1 zV[c,rt,CourseSession,Therapist] <= +2
 triangle1[Role,CourseSession,Instructor,r]: +1 zV[r,rt,Role,CourseSession] +1 zV[r,lb,Role,Instructor]
  +1 zV[r,rt,CourseSession,Instructor] <= +2
 triangle1[Role,CourseSession,Instructor,c]: +1 zV[c,rt,Role,CourseSession] +1 zV[c,lb,Role,Instructor]
  +1 zV[c,rt,CourseSession,Instructor] <= +2
 triangle1[Role,Clinic,Location,r]: +1 zV[r,rt,Role,Clinic] +1 zV[r,lb,Role,Location] +1 zV[r,rt,Clinic,Location]
  <= +2
 triangle1[Role,Clinic,Location,c]: +1 zV[c,rt,Role,Clinic] +1 zV[c,lb,Role,Location] +1 zV[c,rt,Clinic,Location]
  <= +2
 triangle1[Role,Clinic,Therapist,r]: +1 zV[r,rt,Role,Clinic] +1 zV[r,lb,Role,Therapist] +1 zV[r,rt,Clinic,Therapist]
  <= +2
 triangle1[Role,Clinic,Therapist,c]: +1 zV[c,rt,Role,Clinic] +1 zV[c,lb,Role,Therapist] +1 zV[c,rt,Clinic,Therapist]
  <= +2
 triangle1[Role,Clinic,Instructor,r]: +1 zV[r,rt,Role,Clinic] +1 zV[r,lb,Role,Instructor] +1 zV[r,rt,Clinic,Instructor]
  <= +2
 triangle1[Role,Clinic,Instructor,c]: +1 zV[c,rt,Role,Clinic] +1 zV[c,lb,Role,Instructor] +1 zV[c,rt,Clinic,Instructor]
  <= +2
 triangle1[Role,Location,Therapist,r]: +1 zV[r,rt,Role,Location] +1 zV[r,lb,Role,Therapist] +1 zV[r,rt,Location,Therapist]
  <= +2
 triangle1[Role,Location,Therapist,c]: +1 zV[c,rt,Role,Location] +1 zV[c,lb,Role,Therapist] +1 zV[c,rt,Location,Therapist]
  <= +2
 triangle1[Role,Location,Instructor,r]: +1 zV[r,rt,Role,Location] +1 zV[r,lb,Role,Instructor] +1 zV[r,rt,Location,Instructor]
  <= +2
 triangle1[Role,Location,Instructor,c]: +1 zV[c,rt,Role,Location] +1 zV[c,lb,Role,Instructor] +1 zV[c,rt,Location,Instructor]
  <= +2
 triangle1[Role,Therapist,Instructor,r]: +1 zV[r,rt,Role,Therapist] +1 zV[r,lb,Role,Instructor] +1 zV[r,rt,Therapist,Instructor]
  <= +2
 triangle1[Role,Therapist,Instructor,c]: +1 zV[c,rt,Role,Therapist] +1 zV[c,lb,Role,Instructor] +1 zV[c,rt,Therapist,Instructor]
  <= +2
 triangle1[User,CourseSession,Clinic,r]: +1 zV[r,rt,User,CourseSession] +1 zV[r,lb,User,Clinic] +1 zV[r,rt,CourseSession,Clinic]
  <= +2
 triangle1[User,CourseSession,Clinic,c]: +1 zV[c,rt,User,CourseSession] +1 zV[c,lb,User,Clinic] +1 zV[c,rt,CourseSession,Clinic]
  <= +2
 triangle1[User,CourseSession,Location,r]: +1 zV[r,rt,User,CourseSession] +1 zV[r,lb,User,Location] +1 zV[r,rt,CourseSession,Location]
  <= +2
 triangle1[User,CourseSession,Location,c]: +1 zV[c,rt,User,CourseSession] +1 zV[c,lb,User,Location] +1 zV[c,rt,CourseSession,Location]
  <= +2
 triangle1[User,CourseSession,Therapist,r]: +1 zV[r,rt,User,CourseSession] +1 zV[r,lb,User,Therapist]
  +1 zV[r,rt,CourseSession,Therapist] <= +2
 triangle1[User,CourseSession,Therapist,c]: +1 zV[c,rt,User,CourseSession] +1 zV[c,lb,User,Therapist]
  +1 zV[c,rt,CourseSession,Therapist] <= +2
 triangle1[User,CourseSession,Instructor,r]: +1 zV[r,rt,User,CourseSession] +1 zV[r,lb,User,Instructor]
  +1 zV[r,rt,CourseSession,Instructor] <= +2
 triangle1[User,CourseSession,Instructor,c]: +1 zV[c,rt,User,CourseSession] +1 zV[c,lb,User,Instructor]
  +1 zV[c,rt,CourseSession,Instructor] <= +2
 triangle1[User,Clinic,Location,r]: +1 zV[r,rt,User,Clinic] +1 zV[r,lb,User,Location] +1 zV[r,rt,Clinic,Location]
  <= +2
 triangle1[User,Clinic,Location,c]: +1 zV[c,rt,User,Clinic] +1 zV[c,lb,User,Location] +1 zV[c,rt,Clinic,Location]
  <= +2
 triangle1[User,Clinic,Therapist,r]: +1 zV[r,rt,User,Clinic] +1 zV[r,lb,User,Therapist] +1 zV[r,rt,Clinic,Therapist]
  <= +2
 triangle1[User,Clinic,Therapist,c]: +1 zV[c,rt,User,Clinic] +1 zV[c,lb,User,Therapist] +1 zV[c,rt,Clinic,Therapist]
  <= +2
 triangle1[User,Clinic,Instructor,r]: +1 zV[r,rt,User,Clinic] +1 zV[r,lb,User,Instructor] +1 zV[r,rt,Clinic,Instructor]
  <= +2
 triangle1[User,Clinic,Instructor,c]: +1 zV[c,rt,User,Clinic] +1 zV[c,lb,User,Instructor] +1 zV[c,rt,Clinic,Instructor]
  <= +2
 triangle1[User,Location,Therapist,r]: +1 zV[r,rt,User,Location] +1 zV[r,lb,User,Therapist] +1 zV[r,rt,Location,Therapist]
  <= +2
 triangle1[User,Location,Therapist,c]: +1 zV[c,rt,User,Location] +1 zV[c,lb,User,Therapist] +1 zV[c,rt,Location,Therapist]
  <= +2
 triangle1[User,Location,Instructor,r]: +1 zV[r,rt,User,Location] +1 zV[r,lb,User,Instructor] +1 zV[r,rt,Location,Instructor]
  <= +2
 triangle1[User,Location,Instructor,c]: +1 zV[c,rt,User,Location] +1 zV[c,lb,User,Instructor] +1 zV[c,rt,Location,Instructor]
  <= +2
 triangle1[User,Therapist,Instructor,r]: +1 zV[r,rt,User,Therapist] +1 zV[r,lb,User,Instructor] +1 zV[r,rt,Therapist,Instructor]
  <= +2
 triangle1[User,Therapist,Instructor,c]: +1 zV[c,rt,User,Therapist] +1 zV[c,lb,User,Instructor] +1 zV[c,rt,Therapist,Instructor]
  <= +2
 triangle1[CourseSession,Clinic,Location,r]: +1 zV[r,rt,CourseSession,Clinic] +1 zV[r,lb,CourseSession,Location]
  +1 zV[r,rt,Clinic,Location] <= +2
 triangle1[CourseSession,Clinic,Location,c]: +1 zV[c,rt,CourseSession,Clinic] +1 zV[c,lb,CourseSession,Location]
  +1 zV[c,rt,Clinic,Location] <= +2
 triangle1[CourseSession,Clinic,Therapist,r]: +1 zV[r,rt,CourseSession,Clinic] +1 zV[r,lb,CourseSession,Therapist]
  +1 zV[r,rt,Clinic,Therapist] <= +2
 triangle1[CourseSession,Clinic,Therapist,c]: +1 zV[c,rt,CourseSession,Clinic] +1 zV[c,lb,CourseSession,Therapist]
  +1 zV[c,rt,Clinic,Therapist] <= +2
 triangle1[CourseSession,Clinic,Instructor,r]: +1 zV[r,rt,CourseSession,Clinic] +1 zV[r,lb,CourseSession,Instructor]
  +1 zV[r,rt,Clinic,Instructor] <= +2
 triangle1[CourseSession,Clinic,Instructor,c]: +1 zV[c,rt,CourseSession,Clinic] +1 zV[c,lb,CourseSession,Instructor]
  +1 zV[c,rt,Clinic,Instructor] <= +2
 triangle1[CourseSession,Location,Therapist,r]: +1 zV[r,rt,CourseSession,Location] +1 zV[r,lb,CourseSession,Therapist]
  +1 zV[r,rt,Location,Therapist] <= +2
 triangle1[CourseSession,Location,Therapist,c]: +1 zV[c,rt,CourseSession,Location] +1 zV[c,lb,CourseSession,Therapist]
  +1 zV[c,rt,Location,Therapist] <= +2
 triangle1[CourseSession,Location,Instructor,r]: +1 zV[r,rt,CourseSession,Location] +1 zV[r,lb,CourseSession,Instructor]
  +1 zV[r,rt,Location,Instructor] <= +2
 triangle1[CourseSession,Location,Instructor,c]: +1 zV[c,rt,CourseSession,Location] +1 zV[c,lb,CourseSession,Instructor]
  +1 zV[c,rt,Location,Instructor] <= +2
 triangle1[CourseSession,Therapist,Instructor,r]: +1 zV[r,rt,CourseSession,Therapist] +1 zV[r,lb,CourseSession,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[CourseSession,Therapist,Instructor,c]: +1 zV[c,rt,CourseSession,Therapist] +1 zV[c,lb,CourseSession,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle1[Clinic,Location,Therapist,r]: +1 zV[r,rt,Clinic,Location] +1 zV[r,lb,Clinic,Therapist] +1 zV[r,rt,Location,Therapist]
  <= +2
 triangle1[Clinic,Location,Therapist,c]: +1 zV[c,rt,Clinic,Location] +1 zV[c,lb,Clinic,Therapist] +1 zV[c,rt,Location,Therapist]
  <= +2
 triangle1[Clinic,Location,Instructor,r]: +1 zV[r,rt,Clinic,Location] +1 zV[r,lb,Clinic,Instructor] +1 zV[r,rt,Location,Instructor]
  <= +2
 triangle1[Clinic,Location,Instructor,c]: +1 zV[c,rt,Clinic,Location] +1 zV[c,lb,Clinic,Instructor] +1 zV[c,rt,Location,Instructor]
  <= +2
 triangle1[Clinic,Therapist,Instructor,r]: +1 zV[r,rt,Clinic,Therapist] +1 zV[r,lb,Clinic,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[Clinic,Therapist,Instructor,c]: +1 zV[c,rt,Clinic,Therapist] +1 zV[c,lb,Clinic,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle1[Location,Therapist,Instructor,r]: +1 zV[r,rt,Location,Therapist] +1 zV[r,lb,Location,Instructor]
  +1 zV[r,rt,Therapist,Instructor] <= +2
 triangle1[Location,Therapist,Instructor,c]: +1 zV[c,rt,Location,Therapist] +1 zV[c,lb,Location,Instructor]
  +1 zV[c,rt,Therapist,Instructor] <= +2
 triangle2[Client,CertificateTemplate,Course,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Course]
  +1 zV[r,lb,CertificateTemplate,Course] <= +2
 triangle2[Client,CertificateTemplate,Course,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Course]
  +1 zV[c,lb,CertificateTemplate,Course] <= +2
 triangle2[Client,CertificateTemplate,Role,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Role]
  +1 zV[r,lb,CertificateTemplate,Role] <= +2
 triangle2[Client,CertificateTemplate,Role,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Role]
  +1 zV[c,lb,CertificateTemplate,Role] <= +2
 triangle2[Client,CertificateTemplate,User,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,User]
  +1 zV[r,lb,CertificateTemplate,User] <= +2
 triangle2[Client,CertificateTemplate,User,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,User]
  +1 zV[c,lb,CertificateTemplate,User] <= +2
 triangle2[Client,CertificateTemplate,CourseSession,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,CourseSession]
  +1 zV[r,lb,CertificateTemplate,CourseSession] <= +2
 triangle2[Client,CertificateTemplate,CourseSession,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,CourseSession]
  +1 zV[c,lb,CertificateTemplate,CourseSession] <= +2
 triangle2[Client,CertificateTemplate,Clinic,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Clinic]
  +1 zV[r,lb,CertificateTemplate,Clinic] <= +2
 triangle2[Client,CertificateTemplate,Clinic,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Clinic]
  +1 zV[c,lb,CertificateTemplate,Clinic] <= +2
 triangle2[Client,CertificateTemplate,Location,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Location]
  +1 zV[r,lb,CertificateTemplate,Location] <= +2
 triangle2[Client,CertificateTemplate,Location,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Location]
  +1 zV[c,lb,CertificateTemplate,Location] <= +2
 triangle2[Client,CertificateTemplate,Therapist,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Therapist]
  +1 zV[r,lb,CertificateTemplate,Therapist] <= +2
 triangle2[Client,CertificateTemplate,Therapist,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Therapist]
  +1 zV[c,lb,CertificateTemplate,Therapist] <= +2
 triangle2[Client,CertificateTemplate,Instructor,r]: +1 zV[r,lb,Client,CertificateTemplate] +1 zV[r,rt,Client,Instructor]
  +1 zV[r,lb,CertificateTemplate,Instructor] <= +2
 triangle2[Client,CertificateTemplate,Instructor,c]: +1 zV[c,lb,Client,CertificateTemplate] +1 zV[c,rt,Client,Instructor]
  +1 zV[c,lb,CertificateTemplate,Instructor] <= +2
 triangle2[Client,Course,Role,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,Role] +1 zV[r,lb,Course,Role]
  <= +2
 triangle2[Client,Course,Role,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,Role] +1 zV[c,lb,Course,Role]
  <= +2
 triangle2[Client,Course,User,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,User] +1 zV[r,lb,Course,User]
  <= +2
 triangle2[Client,Course,User,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,User] +1 zV[c,lb,Course,User]
  <= +2
 triangle2[Client,Course,CourseSession,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,CourseSession]
  +1 zV[r,lb,Course,CourseSession] <= +2
 triangle2[Client,Course,CourseSession,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,CourseSession]
  +1 zV[c,lb,Course,CourseSession] <= +2
 triangle2[Client,Course,Clinic,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,Course,Clinic]
  <= +2
 triangle2[Client,Course,Clinic,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,Course,Clinic]
  <= +2
 triangle2[Client,Course,Location,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,Location] +1 zV[r,lb,Course,Location]
  <= +2
 triangle2[Client,Course,Location,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,Location] +1 zV[c,lb,Course,Location]
  <= +2
 triangle2[Client,Course,Therapist,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,Course,Therapist]
  <= +2
 triangle2[Client,Course,Therapist,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,Course,Therapist]
  <= +2
 triangle2[Client,Course,Instructor,r]: +1 zV[r,lb,Client,Course] +1 zV[r,rt,Client,Instructor] +1 zV[r,lb,Course,Instructor]
  <= +2
 triangle2[Client,Course,Instructor,c]: +1 zV[c,lb,Client,Course] +1 zV[c,rt,Client,Instructor] +1 zV[c,lb,Course,Instructor]
  <= +2
 triangle2[Client,Role,User,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,User] +1 zV[r,lb,Role,User]
  <= +2
 triangle2[Client,Role,User,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,User] +1 zV[c,lb,Role,User]
  <= +2
 triangle2[Client,Role,CourseSession,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,Role,CourseSession]
  <= +2
 triangle2[Client,Role,CourseSession,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,Role,CourseSession]
  <= +2
 triangle2[Client,Role,Clinic,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,Role,Clinic]
  <= +2
 triangle2[Client,Role,Clinic,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,Role,Clinic]
  <= +2
 triangle2[Client,Role,Location,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,Location] +1 zV[r,lb,Role,Location]
  <= +2
 triangle2[Client,Role,Location,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,Location] +1 zV[c,lb,Role,Location]
  <= +2
 triangle2[Client,Role,Therapist,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,Role,Therapist]
  <= +2
 triangle2[Client,Role,Therapist,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,Role,Therapist]
  <= +2
 triangle2[Client,Role,Instructor,r]: +1 zV[r,lb,Client,Role] +1 zV[r,rt,Client,Instructor] +1 zV[r,lb,Role,Instructor]
  <= +2
 triangle2[Client,Role,Instructor,c]: +1 zV[c,lb,Client,Role] +1 zV[c,rt,Client,Instructor] +1 zV[c,lb,Role,Instructor]
  <= +2
 triangle2[Client,User,CourseSession,r]: +1 zV[r,lb,Client,User] +1 zV[r,rt,Client,CourseSession] +1 zV[r,lb,User,CourseSession]
  <= +2
 triangle2[Client,User,CourseSession,c]: +1 zV[c,lb,Client,User] +1 zV[c,rt,Client,CourseSession] +1 zV[c,lb,User,CourseSession]
  <= +2
 triangle2[Client,User,Clinic,r]: +1 zV[r,lb,Client,User] +1 zV[r,rt,Client,Clinic] +1 zV[r,lb,User,Clinic]
  <= +2
 triangle2[Client,User,Clinic,c]: +1 zV[c,lb,Client,User] +1 zV[c,rt,Client,Clinic] +1 zV[c,lb,User,Clinic]
  <= +2
 triangle2[Client,User,Location,r]: +1 zV[r,lb,Client,User] +1 zV[r,rt,Client,Location] +1 zV[r,lb,User,Location]
  <= +2
 triangle2[Client,User,Location,c]: +1 zV[c,lb,Client,User] +1 zV[c,rt,Client,Location] +1 zV[c,lb,User,Location]
  <= +2
 triangle2[Client,User,Therapist,r]: +1 zV[r,lb,Client,User] +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,User,Therapist]
  <= +2
 triangle2[Client,User,Therapist,c]: +1 zV[c,lb,Client,User] +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,User,Therapist]
  <= +2
 triangle2[Client,User,Instructor,r]: +1 zV[r,lb,Client,User] +1 zV[r,rt,Client,Instructor] +1 zV[r,lb,User,Instructor]
  <= +2
 triangle2[Client,User,Instructor,c]: +1 zV[c,lb,Client,User] +1 zV[c,rt,Client,Instructor] +1 zV[c,lb,User,Instructor]
  <= +2
 triangle2[Client,CourseSession,Clinic,r]: +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,Client,Clinic]
  +1 zV[r,lb,CourseSession,Clinic] <= +2
 triangle2[Client,CourseSession,Clinic,c]: +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,Client,Clinic]
  +1 zV[c,lb,CourseSession,Clinic] <= +2
 triangle2[Client,CourseSession,Location,r]: +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,Client,Location]
  +1 zV[r,lb,CourseSession,Location] <= +2
 triangle2[Client,CourseSession,Location,c]: +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,Client,Location]
  +1 zV[c,lb,CourseSession,Location] <= +2
 triangle2[Client,CourseSession,Therapist,r]: +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,Client,Therapist]
  +1 zV[r,lb,CourseSession,Therapist] <= +2
 triangle2[Client,CourseSession,Therapist,c]: +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,Client,Therapist]
  +1 zV[c,lb,CourseSession,Therapist] <= +2
 triangle2[Client,CourseSession,Instructor,r]: +1 zV[r,lb,Client,CourseSession] +1 zV[r,rt,Client,Instructor]
  +1 zV[r,lb,CourseSession,Instructor] <= +2
 triangle2[Client,CourseSession,Instructor,c]: +1 zV[c,lb,Client,CourseSession] +1 zV[c,rt,Client,Instructor]
  +1 zV[c,lb,CourseSession,Instructor] <= +2
 triangle2[Client,Clinic,Location,r]: +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,Client,Location] +1 zV[r,lb,Clinic,Location]
  <= +2
 triangle2[Client,Clinic,Location,c]: +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,Client,Location] +1 zV[c,lb,Clinic,Location]
  <= +2
 triangle2[Client,Clinic,Therapist,r]: +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,Clinic,Therapist]
  <= +2
 triangle2[Client,Clinic,Therapist,c]: +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,Clinic,Therapist]
  <= +2
 triangle2[Client,Clinic,Instructor,r]: +1 zV[r,lb,Client,Clinic] +1 zV[r,rt,Client,Instructor] +1 zV[r,lb,Clinic,Instructor]
  <= +2
 triangle2[Client,Clinic,Instructor,c]: +1 zV[c,lb,Client,Clinic] +1 zV[c,rt,Client,Instructor] +1 zV[c,lb,Clinic,Instructor]
  <= +2
 triangle2[Client,Location,Therapist,r]: +1 zV[r,lb,Client,Location] +1 zV[r,rt,Client,Therapist] +1 zV[r,lb,Location,Therapist]
  <= +2
 triangle2[Client,Location,Therapist,c]: +1 zV[c,lb,Client,Location] +1 zV[c,rt,Client,Therapist] +1 zV[c,lb,Location,Therapist]
  <= +2
 triangle2[Client,Location,Instructor,r]: +1 zV[r,lb,Client,Location] +1 zV[r,rt,Client,Instructor] +1 zV[r,lb,Location,Instructor]
  <= +2
 triangle2[Client,Location,Instructor,c]: +1 zV[c,lb,Client,Location] +1 zV[c,rt,Client,Instructor] +1 zV[c,lb,Location,Instructor]
  <= +2
 triangle2[Client,Therapist,Instructor,r]: +1 zV[r,lb,Client,Therapist] +1 zV[r,rt,Client,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[Client,Therapist,Instructor,c]: +1 zV[c,lb,Client,Therapist] +1 zV[c,rt,Client,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
 triangle2[CertificateTemplate,Course,Role,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,Role]
  +1 zV[r,lb,Course,Role] <= +2
 triangle2[CertificateTemplate,Course,Role,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Role]
  +1 zV[c,lb,Course,Role] <= +2
 triangle2[CertificateTemplate,Course,User,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,User]
  +1 zV[r,lb,Course,User] <= +2
 triangle2[CertificateTemplate,Course,User,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,User]
  +1 zV[c,lb,Course,User] <= +2
 triangle2[CertificateTemplate,Course,CourseSession,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,Course,CourseSession] <= +2
 triangle2[CertificateTemplate,Course,CourseSession,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,Course,CourseSession] <= +2
 triangle2[CertificateTemplate,Course,Clinic,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,Clinic]
  +1 zV[r,lb,Course,Clinic] <= +2
 triangle2[CertificateTemplate,Course,Clinic,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Clinic]
  +1 zV[c,lb,Course,Clinic] <= +2
 triangle2[CertificateTemplate,Course,Location,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,Location]
  +1 zV[r,lb,Course,Location] <= +2
 triangle2[CertificateTemplate,Course,Location,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Location]
  +1 zV[c,lb,Course,Location] <= +2
 triangle2[CertificateTemplate,Course,Therapist,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,Course,Therapist] <= +2
 triangle2[CertificateTemplate,Course,Therapist,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[c,lb,Course,Therapist] <= +2
 triangle2[CertificateTemplate,Course,Instructor,r]: +1 zV[r,lb,CertificateTemplate,Course] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,Course,Instructor] <= +2
 triangle2[CertificateTemplate,Course,Instructor,c]: +1 zV[c,lb,CertificateTemplate,Course] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,Course,Instructor] <= +2
 triangle2[CertificateTemplate,Role,User,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,User]
  +1 zV[r,lb,Role,User] <= +2
 triangle2[CertificateTemplate,Role,User,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,User]
  +1 zV[c,lb,Role,User] <= +2
 triangle2[CertificateTemplate,Role,CourseSession,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,Role,CourseSession] <= +2
 triangle2[CertificateTemplate,Role,CourseSession,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,Role,CourseSession] <= +2
 triangle2[CertificateTemplate,Role,Clinic,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,Clinic]
  +1 zV[r,lb,Role,Clinic] <= +2
 triangle2[CertificateTemplate,Role,Clinic,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,Clinic]
  +1 zV[c,lb,Role,Clinic] <= +2
 triangle2[CertificateTemplate,Role,Location,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,Location]
  +1 zV[r,lb,Role,Location] <= +2
 triangle2[CertificateTemplate,Role,Location,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,Location]
  +1 zV[c,lb,Role,Location] <= +2
 triangle2[CertificateTemplate,Role,Therapist,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,Role,Therapist] <= +2
 triangle2[CertificateTemplate,Role,Therapist,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[c,lb,Role,Therapist] <= +2
 triangle2[CertificateTemplate,Role,Instructor,r]: +1 zV[r,lb,CertificateTemplate,Role] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,Role,Instructor] <= +2
 triangle2[CertificateTemplate,Role,Instructor,c]: +1 zV[c,lb,CertificateTemplate,Role] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,Role,Instructor] <= +2
 triangle2[CertificateTemplate,User,CourseSession,r]: +1 zV[r,lb,CertificateTemplate,User] +1 zV[r,rt,CertificateTemplate,CourseSession]
  +1 zV[r,lb,User,CourseSession] <= +2
 triangle2[CertificateTemplate,User,CourseSession,c]: +1 zV[c,lb,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,CourseSession]
  +1 zV[c,lb,User,CourseSession] <= +2
 triangle2[CertificateTemplate,User,Clinic,r]: +1 zV[r,lb,CertificateTemplate,User] +1 zV[r,rt,CertificateTemplate,Clinic]
  +1 zV[r,lb,User,Clinic] <= +2
 triangle2[CertificateTemplate,User,Clinic,c]: +1 zV[c,lb,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,Clinic]
  +1 zV[c,lb,User,Clinic] <= +2
 triangle2[CertificateTemplate,User,Location,r]: +1 zV[r,lb,CertificateTemplate,User] +1 zV[r,rt,CertificateTemplate,Location]
  +1 zV[r,lb,User,Location] <= +2
 triangle2[CertificateTemplate,User,Location,c]: +1 zV[c,lb,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,Location]
  +1 zV[c,lb,User,Location] <= +2
 triangle2[CertificateTemplate,User,Therapist,r]: +1 zV[r,lb,CertificateTemplate,User] +1 zV[r,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,User,Therapist] <= +2
 triangle2[CertificateTemplate,User,Therapist,c]: +1 zV[c,lb,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[c,lb,User,Therapist] <= +2
 triangle2[CertificateTemplate,User,Instructor,r]: +1 zV[r,lb,CertificateTemplate,User] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,User,Instructor] <= +2
 triangle2[CertificateTemplate,User,Instructor,c]: +1 zV[c,lb,CertificateTemplate,User] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,User,Instructor] <= +2
 triangle2[CertificateTemplate,CourseSession,Clinic,r]: +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,CertificateTemplate,Clinic] +1 zV[r,lb,CourseSession,Clinic] <= +2
 triangle2[CertificateTemplate,CourseSession,Clinic,c]: +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,CertificateTemplate,Clinic] +1 zV[c,lb,CourseSession,Clinic] <= +2
 triangle2[CertificateTemplate,CourseSession,Location,r]: +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,CertificateTemplate,Location] +1 zV[r,lb,CourseSession,Location] <= +2
 triangle2[CertificateTemplate,CourseSession,Location,c]: +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,CertificateTemplate,Location] +1 zV[c,lb,CourseSession,Location] <= +2
 triangle2[CertificateTemplate,CourseSession,Therapist,r]: +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,CertificateTemplate,Therapist] +1 zV[r,lb,CourseSession,Therapist] <= +2
 triangle2[CertificateTemplate,CourseSession,Therapist,c]: +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,CertificateTemplate,Therapist] +1 zV[c,lb,CourseSession,Therapist] <= +2
 triangle2[CertificateTemplate,CourseSession,Instructor,r]: +1 zV[r,lb,CertificateTemplate,CourseSession]
  +1 zV[r,rt,CertificateTemplate,Instructor] +1 zV[r,lb,CourseSession,Instructor] <= +2
 triangle2[CertificateTemplate,CourseSession,Instructor,c]: +1 zV[c,lb,CertificateTemplate,CourseSession]
  +1 zV[c,rt,CertificateTemplate,Instructor] +1 zV[c,lb,CourseSession,Instructor] <= +2
 triangle2[CertificateTemplate,Clinic,Location,r]: +1 zV[r,lb,CertificateTemplate,Clinic] +1 zV[r,rt,CertificateTemplate,Location]
  +1 zV[r,lb,Clinic,Location] <= +2
 triangle2[CertificateTemplate,Clinic,Location,c]: +1 zV[c,lb,CertificateTemplate,Clinic] +1 zV[c,rt,CertificateTemplate,Location]
  +1 zV[c,lb,Clinic,Location] <= +2
 triangle2[CertificateTemplate,Clinic,Therapist,r]: +1 zV[r,lb,CertificateTemplate,Clinic] +1 zV[r,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,Clinic,Therapist] <= +2
 triangle2[CertificateTemplate,Clinic,Therapist,c]: +1 zV[c,lb,CertificateTemplate,Clinic] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[c,lb,Clinic,Therapist] <= +2
 triangle2[CertificateTemplate,Clinic,Instructor,r]: +1 zV[r,lb,CertificateTemplate,Clinic] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,Clinic,Instructor] <= +2
 triangle2[CertificateTemplate,Clinic,Instructor,c]: +1 zV[c,lb,CertificateTemplate,Clinic] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,Clinic,Instructor] <= +2
 triangle2[CertificateTemplate,Location,Therapist,r]: +1 zV[r,lb,CertificateTemplate,Location] +1 zV[r,rt,CertificateTemplate,Therapist]
  +1 zV[r,lb,Location,Therapist] <= +2
 triangle2[CertificateTemplate,Location,Therapist,c]: +1 zV[c,lb,CertificateTemplate,Location] +1 zV[c,rt,CertificateTemplate,Therapist]
  +1 zV[c,lb,Location,Therapist] <= +2
 triangle2[CertificateTemplate,Location,Instructor,r]: +1 zV[r,lb,CertificateTemplate,Location] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,Location,Instructor] <= +2
 triangle2[CertificateTemplate,Location,Instructor,c]: +1 zV[c,lb,CertificateTemplate,Location] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,Location,Instructor] <= +2
 triangle2[CertificateTemplate,Therapist,Instructor,r]: +1 zV[r,lb,CertificateTemplate,Therapist] +1 zV[r,rt,CertificateTemplate,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[CertificateTemplate,Therapist,Instructor,c]: +1 zV[c,lb,CertificateTemplate,Therapist] +1 zV[c,rt,CertificateTemplate,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
 triangle2[Course,Role,User,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,User] +1 zV[r,lb,Role,User]
  <= +2
 triangle2[Course,Role,User,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,User] +1 zV[c,lb,Role,User]
  <= +2
 triangle2[Course,Role,CourseSession,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,Role,CourseSession]
  <= +2
 triangle2[Course,Role,CourseSession,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,Role,CourseSession]
  <= +2
 triangle2[Course,Role,Clinic,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,Clinic] +1 zV[r,lb,Role,Clinic]
  <= +2
 triangle2[Course,Role,Clinic,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,Clinic] +1 zV[c,lb,Role,Clinic]
  <= +2
 triangle2[Course,Role,Location,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,Location] +1 zV[r,lb,Role,Location]
  <= +2
 triangle2[Course,Role,Location,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,Location] +1 zV[c,lb,Role,Location]
  <= +2
 triangle2[Course,Role,Therapist,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,Therapist] +1 zV[r,lb,Role,Therapist]
  <= +2
 triangle2[Course,Role,Therapist,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,Therapist] +1 zV[c,lb,Role,Therapist]
  <= +2
 triangle2[Course,Role,Instructor,r]: +1 zV[r,lb,Course,Role] +1 zV[r,rt,Course,Instructor] +1 zV[r,lb,Role,Instructor]
  <= +2
 triangle2[Course,Role,Instructor,c]: +1 zV[c,lb,Course,Role] +1 zV[c,rt,Course,Instructor] +1 zV[c,lb,Role,Instructor]
  <= +2
 triangle2[Course,User,CourseSession,r]: +1 zV[r,lb,Course,User] +1 zV[r,rt,Course,CourseSession] +1 zV[r,lb,User,CourseSession]
  <= +2
 triangle2[Course,User,CourseSession,c]: +1 zV[c,lb,Course,User] +1 zV[c,rt,Course,CourseSession] +1 zV[c,lb,User,CourseSession]
  <= +2
 triangle2[Course,User,Clinic,r]: +1 zV[r,lb,Course,User] +1 zV[r,rt,Course,Clinic] +1 zV[r,lb,User,Clinic]
  <= +2
 triangle2[Course,User,Clinic,c]: +1 zV[c,lb,Course,User] +1 zV[c,rt,Course,Clinic] +1 zV[c,lb,User,Clinic]
  <= +2
 triangle2[Course,User,Location,r]: +1 zV[r,lb,Course,User] +1 zV[r,rt,Course,Location] +1 zV[r,lb,User,Location]
  <= +2
 triangle2[Course,User,Location,c]: +1 zV[c,lb,Course,User] +1 zV[c,rt,Course,Location] +1 zV[c,lb,User,Location]
  <= +2
 triangle2[Course,User,Therapist,r]: +1 zV[r,lb,Course,User] +1 zV[r,rt,Course,Therapist] +1 zV[r,lb,User,Therapist]
  <= +2
 triangle2[Course,User,Therapist,c]: +1 zV[c,lb,Course,User] +1 zV[c,rt,Course,Therapist] +1 zV[c,lb,User,Therapist]
  <= +2
 triangle2[Course,User,Instructor,r]: +1 zV[r,lb,Course,User] +1 zV[r,rt,Course,Instructor] +1 zV[r,lb,User,Instructor]
  <= +2
 triangle2[Course,User,Instructor,c]: +1 zV[c,lb,Course,User] +1 zV[c,rt,Course,Instructor] +1 zV[c,lb,User,Instructor]
  <= +2
 triangle2[Course,CourseSession,Clinic,r]: +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,Course,Clinic]
  +1 zV[r,lb,CourseSession,Clinic] <= +2
 triangle2[Course,CourseSession,Clinic,c]: +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,Course,Clinic]
  +1 zV[c,lb,CourseSession,Clinic] <= +2
 triangle2[Course,CourseSession,Location,r]: +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,Course,Location]
  +1 zV[r,lb,CourseSession,Location] <= +2
 triangle2[Course,CourseSession,Location,c]: +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,Course,Location]
  +1 zV[c,lb,CourseSession,Location] <= +2
 triangle2[Course,CourseSession,Therapist,r]: +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,Course,Therapist]
  +1 zV[r,lb,CourseSession,Therapist] <= +2
 triangle2[Course,CourseSession,Therapist,c]: +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,Course,Therapist]
  +1 zV[c,lb,CourseSession,Therapist] <= +2
 triangle2[Course,CourseSession,Instructor,r]: +1 zV[r,lb,Course,CourseSession] +1 zV[r,rt,Course,Instructor]
  +1 zV[r,lb,CourseSession,Instructor] <= +2
 triangle2[Course,CourseSession,Instructor,c]: +1 zV[c,lb,Course,CourseSession] +1 zV[c,rt,Course,Instructor]
  +1 zV[c,lb,CourseSession,Instructor] <= +2
 triangle2[Course,Clinic,Location,r]: +1 zV[r,lb,Course,Clinic] +1 zV[r,rt,Course,Location] +1 zV[r,lb,Clinic,Location]
  <= +2
 triangle2[Course,Clinic,Location,c]: +1 zV[c,lb,Course,Clinic] +1 zV[c,rt,Course,Location] +1 zV[c,lb,Clinic,Location]
  <= +2
 triangle2[Course,Clinic,Therapist,r]: +1 zV[r,lb,Course,Clinic] +1 zV[r,rt,Course,Therapist] +1 zV[r,lb,Clinic,Therapist]
  <= +2
 triangle2[Course,Clinic,Therapist,c]: +1 zV[c,lb,Course,Clinic] +1 zV[c,rt,Course,Therapist] +1 zV[c,lb,Clinic,Therapist]
  <= +2
 triangle2[Course,Clinic,Instructor,r]: +1 zV[r,lb,Course,Clinic] +1 zV[r,rt,Course,Instructor] +1 zV[r,lb,Clinic,Instructor]
  <= +2
 triangle2[Course,Clinic,Instructor,c]: +1 zV[c,lb,Course,Clinic] +1 zV[c,rt,Course,Instructor] +1 zV[c,lb,Clinic,Instructor]
  <= +2
 triangle2[Course,Location,Therapist,r]: +1 zV[r,lb,Course,Location] +1 zV[r,rt,Course,Therapist] +1 zV[r,lb,Location,Therapist]
  <= +2
 triangle2[Course,Location,Therapist,c]: +1 zV[c,lb,Course,Location] +1 zV[c,rt,Course,Therapist] +1 zV[c,lb,Location,Therapist]
  <= +2
 triangle2[Course,Location,Instructor,r]: +1 zV[r,lb,Course,Location] +1 zV[r,rt,Course,Instructor] +1 zV[r,lb,Location,Instructor]
  <= +2
 triangle2[Course,Location,Instructor,c]: +1 zV[c,lb,Course,Location] +1 zV[c,rt,Course,Instructor] +1 zV[c,lb,Location,Instructor]
  <= +2
 triangle2[Course,Therapist,Instructor,r]: +1 zV[r,lb,Course,Therapist] +1 zV[r,rt,Course,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[Course,Therapist,Instructor,c]: +1 zV[c,lb,Course,Therapist] +1 zV[c,rt,Course,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
 triangle2[Role,User,CourseSession,r]: +1 zV[r,lb,Role,User] +1 zV[r,rt,Role,CourseSession] +1 zV[r,lb,User,CourseSession]
  <= +2
 triangle2[Role,User,CourseSession,c]: +1 zV[c,lb,Role,User] +1 zV[c,rt,Role,CourseSession] +1 zV[c,lb,User,CourseSession]
  <= +2
 triangle2[Role,User,Clinic,r]: +1 zV[r,lb,Role,User] +1 zV[r,rt,Role,Clinic] +1 zV[r,lb,User,Clinic]
  <= +2
 triangle2[Role,User,Clinic,c]: +1 zV[c,lb,Role,User] +1 zV[c,rt,Role,Clinic] +1 zV[c,lb,User,Clinic]
  <= +2
 triangle2[Role,User,Location,r]: +1 zV[r,lb,Role,User] +1 zV[r,rt,Role,Location] +1 zV[r,lb,User,Location]
  <= +2
 triangle2[Role,User,Location,c]: +1 zV[c,lb,Role,User] +1 zV[c,rt,Role,Location] +1 zV[c,lb,User,Location]
  <= +2
 triangle2[Role,User,Therapist,r]: +1 zV[r,lb,Role,User] +1 zV[r,rt,Role,Therapist] +1 zV[r,lb,User,Therapist]
  <= +2
 triangle2[Role,User,Therapist,c]: +1 zV[c,lb,Role,User] +1 zV[c,rt,Role,Therapist] +1 zV[c,lb,User,Therapist]
  <= +2
 triangle2[Role,User,Instructor,r]: +1 zV[r,lb,Role,User] +1 zV[r,rt,Role,Instructor] +1 zV[r,lb,User,Instructor]
  <= +2
 triangle2[Role,User,Instructor,c]: +1 zV[c,lb,Role,User] +1 zV[c,rt,Role,Instructor] +1 zV[c,lb,User,Instructor]
  <= +2
 triangle2[Role,CourseSession,Clinic,r]: +1 zV[r,lb,Role,CourseSession] +1 zV[r,rt,Role,Clinic] +1 zV[r,lb,CourseSession,Clinic]
  <= +2
 triangle2[Role,CourseSession,Clinic,c]: +1 zV[c,lb,Role,CourseSession] +1 zV[c,rt,Role,Clinic] +1 zV[c,lb,CourseSession,Clinic]
  <= +2
 triangle2[Role,CourseSession,Location,r]: +1 zV[r,lb,Role,CourseSession] +1 zV[r,rt,Role,Location] +1 zV[r,lb,CourseSession,Location]
  <= +2
 triangle2[Role,CourseSession,Location,c]: +1 zV[c,lb,Role,CourseSession] +1 zV[c,rt,Role,Location] +1 zV[c,lb,CourseSession,Location]
  <= +2
 triangle2[Role,CourseSession,Therapist,r]: +1 zV[r,lb,Role,CourseSession] +1 zV[r,rt,Role,Therapist]
  +1 zV[r,lb,CourseSession,Therapist] <= +2
 triangle2[Role,CourseSession,Therapist,c]: +1 zV[c,lb,Role,CourseSession] +1 zV[c,rt,Role,Therapist]
  +1 zV[c,lb,CourseSession,Therapist] <= +2
 triangle2[Role,CourseSession,Instructor,r]: +1 zV[r,lb,Role,CourseSession] +1 zV[r,rt,Role,Instructor]
  +1 zV[r,lb,CourseSession,Instructor] <= +2
 triangle2[Role,CourseSession,Instructor,c]: +1 zV[c,lb,Role,CourseSession] +1 zV[c,rt,Role,Instructor]
  +1 zV[c,lb,CourseSession,Instructor] <= +2
 triangle2[Role,Clinic,Location,r]: +1 zV[r,lb,Role,Clinic] +1 zV[r,rt,Role,Location] +1 zV[r,lb,Clinic,Location]
  <= +2
 triangle2[Role,Clinic,Location,c]: +1 zV[c,lb,Role,Clinic] +1 zV[c,rt,Role,Location] +1 zV[c,lb,Clinic,Location]
  <= +2
 triangle2[Role,Clinic,Therapist,r]: +1 zV[r,lb,Role,Clinic] +1 zV[r,rt,Role,Therapist] +1 zV[r,lb,Clinic,Therapist]
  <= +2
 triangle2[Role,Clinic,Therapist,c]: +1 zV[c,lb,Role,Clinic] +1 zV[c,rt,Role,Therapist] +1 zV[c,lb,Clinic,Therapist]
  <= +2
 triangle2[Role,Clinic,Instructor,r]: +1 zV[r,lb,Role,Clinic] +1 zV[r,rt,Role,Instructor] +1 zV[r,lb,Clinic,Instructor]
  <= +2
 triangle2[Role,Clinic,Instructor,c]: +1 zV[c,lb,Role,Clinic] +1 zV[c,rt,Role,Instructor] +1 zV[c,lb,Clinic,Instructor]
  <= +2
 triangle2[Role,Location,Therapist,r]: +1 zV[r,lb,Role,Location] +1 zV[r,rt,Role,Therapist] +1 zV[r,lb,Location,Therapist]
  <= +2
 triangle2[Role,Location,Therapist,c]: +1 zV[c,lb,Role,Location] +1 zV[c,rt,Role,Therapist] +1 zV[c,lb,Location,Therapist]
  <= +2
 triangle2[Role,Location,Instructor,r]: +1 zV[r,lb,Role,Location] +1 zV[r,rt,Role,Instructor] +1 zV[r,lb,Location,Instructor]
  <= +2
 triangle2[Role,Location,Instructor,c]: +1 zV[c,lb,Role,Location] +1 zV[c,rt,Role,Instructor] +1 zV[c,lb,Location,Instructor]
  <= +2
 triangle2[Role,Therapist,Instructor,r]: +1 zV[r,lb,Role,Therapist] +1 zV[r,rt,Role,Instructor] +1 zV[r,lb,Therapist,Instructor]
  <= +2
 triangle2[Role,Therapist,Instructor,c]: +1 zV[c,lb,Role,Therapist] +1 zV[c,rt,Role,Instructor] +1 zV[c,lb,Therapist,Instructor]
  <= +2
 triangle2[User,CourseSession,Clinic,r]: +1 zV[r,lb,User,CourseSession] +1 zV[r,rt,User,Clinic] +1 zV[r,lb,CourseSession,Clinic]
  <= +2
 triangle2[User,CourseSession,Clinic,c]: +1 zV[c,lb,User,CourseSession] +1 zV[c,rt,User,Clinic] +1 zV[c,lb,CourseSession,Clinic]
  <= +2
 triangle2[User,CourseSession,Location,r]: +1 zV[r,lb,User,CourseSession] +1 zV[r,rt,User,Location] +1 zV[r,lb,CourseSession,Location]
  <= +2
 triangle2[User,CourseSession,Location,c]: +1 zV[c,lb,User,CourseSession] +1 zV[c,rt,User,Location] +1 zV[c,lb,CourseSession,Location]
  <= +2
 triangle2[User,CourseSession,Therapist,r]: +1 zV[r,lb,User,CourseSession] +1 zV[r,rt,User,Therapist]
  +1 zV[r,lb,CourseSession,Therapist] <= +2
 triangle2[User,CourseSession,Therapist,c]: +1 zV[c,lb,User,CourseSession] +1 zV[c,rt,User,Therapist]
  +1 zV[c,lb,CourseSession,Therapist] <= +2
 triangle2[User,CourseSession,Instructor,r]: +1 zV[r,lb,User,CourseSession] +1 zV[r,rt,User,Instructor]
  +1 zV[r,lb,CourseSession,Instructor] <= +2
 triangle2[User,CourseSession,Instructor,c]: +1 zV[c,lb,User,CourseSession] +1 zV[c,rt,User,Instructor]
  +1 zV[c,lb,CourseSession,Instructor] <= +2
 triangle2[User,Clinic,Location,r]: +1 zV[r,lb,User,Clinic] +1 zV[r,rt,User,Location] +1 zV[r,lb,Clinic,Location]
  <= +2
 triangle2[User,Clinic,Location,c]: +1 zV[c,lb,User,Clinic] +1 zV[c,rt,User,Location] +1 zV[c,lb,Clinic,Location]
  <= +2
 triangle2[User,Clinic,Therapist,r]: +1 zV[r,lb,User,Clinic] +1 zV[r,rt,User,Therapist] +1 zV[r,lb,Clinic,Therapist]
  <= +2
 triangle2[User,Clinic,Therapist,c]: +1 zV[c,lb,User,Clinic] +1 zV[c,rt,User,Therapist] +1 zV[c,lb,Clinic,Therapist]
  <= +2
 triangle2[User,Clinic,Instructor,r]: +1 zV[r,lb,User,Clinic] +1 zV[r,rt,User,Instructor] +1 zV[r,lb,Clinic,Instructor]
  <= +2
 triangle2[User,Clinic,Instructor,c]: +1 zV[c,lb,User,Clinic] +1 zV[c,rt,User,Instructor] +1 zV[c,lb,Clinic,Instructor]
  <= +2
 triangle2[User,Location,Therapist,r]: +1 zV[r,lb,User,Location] +1 zV[r,rt,User,Therapist] +1 zV[r,lb,Location,Therapist]
  <= +2
 triangle2[User,Location,Therapist,c]: +1 zV[c,lb,User,Location] +1 zV[c,rt,User,Therapist] +1 zV[c,lb,Location,Therapist]
  <= +2
 triangle2[User,Location,Instructor,r]: +1 zV[r,lb,User,Location] +1 zV[r,rt,User,Instructor] +1 zV[r,lb,Location,Instructor]
  <= +2
 triangle2[User,Location,Instructor,c]: +1 zV[c,lb,User,Location] +1 zV[c,rt,User,Instructor] +1 zV[c,lb,Location,Instructor]
  <= +2
 triangle2[User,Therapist,Instructor,r]: +1 zV[r,lb,User,Therapist] +1 zV[r,rt,User,Instructor] +1 zV[r,lb,Therapist,Instructor]
  <= +2
 triangle2[User,Therapist,Instructor,c]: +1 zV[c,lb,User,Therapist] +1 zV[c,rt,User,Instructor] +1 zV[c,lb,Therapist,Instructor]
  <= +2
 triangle2[CourseSession,Clinic,Location,r]: +1 zV[r,lb,CourseSession,Clinic] +1 zV[r,rt,CourseSession,Location]
  +1 zV[r,lb,Clinic,Location] <= +2
 triangle2[CourseSession,Clinic,Location,c]: +1 zV[c,lb,CourseSession,Clinic] +1 zV[c,rt,CourseSession,Location]
  +1 zV[c,lb,Clinic,Location] <= +2
 triangle2[CourseSession,Clinic,Therapist,r]: +1 zV[r,lb,CourseSession,Clinic] +1 zV[r,rt,CourseSession,Therapist]
  +1 zV[r,lb,Clinic,Therapist] <= +2
 triangle2[CourseSession,Clinic,Therapist,c]: +1 zV[c,lb,CourseSession,Clinic] +1 zV[c,rt,CourseSession,Therapist]
  +1 zV[c,lb,Clinic,Therapist] <= +2
 triangle2[CourseSession,Clinic,Instructor,r]: +1 zV[r,lb,CourseSession,Clinic] +1 zV[r,rt,CourseSession,Instructor]
  +1 zV[r,lb,Clinic,Instructor] <= +2
 triangle2[CourseSession,Clinic,Instructor,c]: +1 zV[c,lb,CourseSession,Clinic] +1 zV[c,rt,CourseSession,Instructor]
  +1 zV[c,lb,Clinic,Instructor] <= +2
 triangle2[CourseSession,Location,Therapist,r]: +1 zV[r,lb,CourseSession,Location] +1 zV[r,rt,CourseSession,Therapist]
  +1 zV[r,lb,Location,Therapist] <= +2
 triangle2[CourseSession,Location,Therapist,c]: +1 zV[c,lb,CourseSession,Location] +1 zV[c,rt,CourseSession,Therapist]
  +1 zV[c,lb,Location,Therapist] <= +2
 triangle2[CourseSession,Location,Instructor,r]: +1 zV[r,lb,CourseSession,Location] +1 zV[r,rt,CourseSession,Instructor]
  +1 zV[r,lb,Location,Instructor] <= +2
 triangle2[CourseSession,Location,Instructor,c]: +1 zV[c,lb,CourseSession,Location] +1 zV[c,rt,CourseSession,Instructor]
  +1 zV[c,lb,Location,Instructor] <= +2
 triangle2[CourseSession,Therapist,Instructor,r]: +1 zV[r,lb,CourseSession,Therapist] +1 zV[r,rt,CourseSession,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[CourseSession,Therapist,Instructor,c]: +1 zV[c,lb,CourseSession,Therapist] +1 zV[c,rt,CourseSession,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
 triangle2[Clinic,Location,Therapist,r]: +1 zV[r,lb,Clinic,Location] +1 zV[r,rt,Clinic,Therapist] +1 zV[r,lb,Location,Therapist]
  <= +2
 triangle2[Clinic,Location,Therapist,c]: +1 zV[c,lb,Clinic,Location] +1 zV[c,rt,Clinic,Therapist] +1 zV[c,lb,Location,Therapist]
  <= +2
 triangle2[Clinic,Location,Instructor,r]: +1 zV[r,lb,Clinic,Location] +1 zV[r,rt,Clinic,Instructor] +1 zV[r,lb,Location,Instructor]
  <= +2
 triangle2[Clinic,Location,Instructor,c]: +1 zV[c,lb,Clinic,Location] +1 zV[c,rt,Clinic,Instructor] +1 zV[c,lb,Location,Instructor]
  <= +2
 triangle2[Clinic,Therapist,Instructor,r]: +1 zV[r,lb,Clinic,Therapist] +1 zV[r,rt,Clinic,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[Clinic,Therapist,Instructor,c]: +1 zV[c,lb,Clinic,Therapist] +1 zV[c,rt,Clinic,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
 triangle2[Location,Therapist,Instructor,r]: +1 zV[r,lb,Location,Therapist] +1 zV[r,rt,Location,Instructor]
  +1 zV[r,lb,Therapist,Instructor] <= +2
 triangle2[Location,Therapist,Instructor,c]: +1 zV[c,lb,Location,Therapist] +1 zV[c,rt,Location,Instructor]
  +1 zV[c,lb,Therapist,Instructor] <= +2
Bounds
 0 <= zV[r,lb,Therapist,Instructor] <= 1
 0 <= zV[c,rt,Therapist,Instructor] <= 1
 0 <= zV[r,rt,Therapist,Instructor] <= 1
 0 <= zV[c,lb,Location,Instructor] <= 1
 0 <= zV[r,lb,Location,Instructor] <= 1
 0 <= zV[c,rt,Location,Instructor] <= 1
 0 <= zV[r,rt,Location,Instructor] <= 1
 0 <= zV[c,lb,Location,Therapist] <= 1
 0 <= zV[r,lb,Location,Therapist] <= 1
 0 <= zV[c,rt,Location,Therapist] <= 1
 0 <= zV[r,rt,Location,Therapist] <= 1
 0 <= zV[c,lb,Clinic,Instructor] <= 1
 0 <= zV[r,lb,Clinic,Instructor] <= 1
 0 <= zV[c,rt,Clinic,Instructor] <= 1
 0 <= zV[r,rt,Clinic,Instructor] <= 1
 0 <= zV[c,lb,Clinic,Therapist] <= 1
 0 <= zV[r,lb,Clinic,Therapist] <= 1
 0 <= zV[c,rt,Clinic,Therapist] <= 1
 0 <= zV[r,rt,Clinic,Therapist] <= 1
 0 <= zV[c,lb,Clinic,Location] <= 1
 0 <= zV[r,lb,Clinic,Location] <= 1
 0 <= zV[r,rt,Client,CertificateTemplate] <= 1
 0 <= zV[c,rt,Client,CertificateTemplate] <= 1
 0 <= zV[r,lb,Client,CertificateTemplate] <= 1
 0 <= zV[c,lb,Client,CertificateTemplate] <= 1
 0 <= zV[r,rt,Client,Course] <= 1
 0 <= zV[c,rt,Client,Course] <= 1
 0 <= zV[r,lb,Client,Course] <= 1
 0 <= zV[c,lb,Client,Course] <= 1
 0 <= zV[r,rt,Client,Role] <= 1
 0 <= zV[c,rt,Client,Role] <= 1
 0 <= zV[r,lb,Client,Role] <= 1
 0 <= zV[c,lb,Client,Role] <= 1
 0 <= zV[r,rt,Client,User] <= 1
 0 <= zV[c,rt,Client,User] <= 1
 0 <= zV[r,lb,Client,User] <= 1
 0 <= zV[c,lb,Client,User] <= 1
 0 <= zV[r,rt,Client,CourseSession] <= 1
 0 <= zV[c,rt,Client,CourseSession] <= 1
 0 <= zV[r,lb,Client,CourseSession] <= 1
 0 <= zV[c,lb,Client,CourseSession] <= 1
 0 <= zV[r,rt,Client,Clinic] <= 1
 0 <= zV[c,rt,Client,Clinic] <= 1
 0 <= zV[r,lb,Client,Clinic] <= 1
 0 <= zV[c,lb,Client,Clinic] <= 1
 0 <= zV[r,rt,Client,Location] <= 1
 0 <= zV[c,rt,Client,Location] <= 1
 0 <= zV[r,lb,Client,Location] <= 1
 0 <= zV[c,lb,Client,Location] <= 1
 0 <= zV[r,rt,Client,Therapist] <= 1
 0 <= zV[c,rt,Client,Therapist] <= 1
 0 <= zV[r,lb,Client,Therapist] <= 1
 0 <= zV[c,lb,Client,Therapist] <= 1
 0 <= zV[r,rt,Client,Instructor] <= 1
 0 <= zV[c,rt,Client,Instructor] <= 1
 0 <= zV[r,lb,Client,Instructor] <= 1
 0 <= zV[c,lb,Client,Instructor] <= 1
 0 <= zV[r,rt,CertificateTemplate,Course] <= 1
 0 <= zV[c,rt,CertificateTemplate,Course] <= 1
 0 <= zV[r,lb,CertificateTemplate,Course] <= 1
 0 <= zV[c,lb,CertificateTemplate,Course] <= 1
 0 <= zV[r,rt,CertificateTemplate,Role] <= 1
 0 <= zV[c,rt,CertificateTemplate,Role] <= 1
 0 <= zV[r,lb,CertificateTemplate,Role] <= 1
 0 <= zV[c,lb,CertificateTemplate,Role] <= 1
 0 <= zV[r,rt,CertificateTemplate,User] <= 1
 0 <= zV[c,rt,CertificateTemplate,User] <= 1
 0 <= zV[r,lb,CertificateTemplate,User] <= 1
 0 <= zV[c,lb,CertificateTemplate,User] <= 1
 0 <= zV[r,rt,CertificateTemplate,CourseSession] <= 1
 0 <= zV[c,rt,CertificateTemplate,CourseSession] <= 1
 0 <= zV[r,lb,CertificateTemplate,CourseSession] <= 1
 0 <= zV[c,lb,CertificateTemplate,CourseSession] <= 1
 0 <= zV[r,rt,CertificateTemplate,Clinic] <= 1
 0 <= zV[c,rt,CertificateTemplate,Clinic] <= 1
 0 <= zV[r,lb,CertificateTemplate,Clinic] <= 1
 0 <= zV[c,lb,CertificateTemplate,Clinic] <= 1
 0 <= zV[r,rt,CertificateTemplate,Location] <= 1
 0 <= zV[c,rt,CertificateTemplate,Location] <= 1
 0 <= zV[r,lb,CertificateTemplate,Location] <= 1
 0 <= zV[c,lb,CertificateTemplate,Location] <= 1
 0 <= zV[r,rt,CertificateTemplate,Therapist] <= 1
 0 <= zV[c,rt,CertificateTemplate,Therapist] <= 1
 0 <= zV[r,lb,CertificateTemplate,Therapist] <= 1
 0 <= zV[c,lb,CertificateTemplate,Therapist] <= 1
 0 <= zV[r,rt,CertificateTemplate,Instructor] <= 1
 0 <= zV[c,rt,CertificateTemplate,Instructor] <= 1
 0 <= zV[r,lb,CertificateTemplate,Instructor] <= 1
 0 <= zV[c,lb,CertificateTemplate,Instructor] <= 1
 0 <= zV[r,rt,Course,Role] <= 1
 0 <= zV[c,rt,Course,Role] <= 1
 0 <= zV[r,lb,Course,Role] <= 1
 0 <= zV[c,lb,Course,Role] <= 1
 0 <= zV[r,rt,Course,User] <= 1
 0 <= zV[c,rt,Course,User] <= 1
 0 <= zV[r,lb,Course,User] <= 1
 0 <= zV[c,lb,Course,User] <= 1
 0 <= zV[r,rt,Course,CourseSession] <= 1
 0 <= zV[c,rt,Course,CourseSession] <= 1
 0 <= zV[r,lb,Course,CourseSession] <= 1
 0 <= zV[c,lb,Course,CourseSession] <= 1
 0 <= zV[r,rt,Course,Clinic] <= 1
 0 <= zV[c,rt,Course,Clinic] <= 1
 0 <= zV[r,lb,Course,Clinic] <= 1
 0 <= zV[c,lb,Course,Clinic] <= 1
 0 <= zV[r,rt,Course,Location] <= 1
 0 <= zV[c,rt,Course,Location] <= 1
 0 <= zV[r,lb,Course,Location] <= 1
 0 <= zV[c,lb,Course,Location] <= 1
 0 <= zV[r,rt,Course,Therapist] <= 1
 0 <= zV[c,rt,Course,Therapist] <= 1
 0 <= zV[r,lb,Course,Therapist] <= 1
 0 <= zV[c,lb,Course,Therapist] <= 1
 0 <= zV[r,rt,Course,Instructor] <= 1
 0 <= zV[c,rt,Course,Instructor] <= 1
 0 <= zV[r,lb,Course,Instructor] <= 1
 0 <= zV[c,lb,Course,Instructor] <= 1
 0 <= zV[r,rt,Role,User] <= 1
 0 <= zV[c,rt,Role,User] <= 1
 0 <= zV[r,lb,Role,User] <= 1
 0 <= zV[c,lb,Role,User] <= 1
 0 <= zV[r,rt,Role,CourseSession] <= 1
 0 <= zV[c,rt,Role,CourseSession] <= 1
 0 <= zV[r,lb,Role,CourseSession] <= 1
 0 <= zV[c,lb,Role,CourseSession] <= 1
 0 <= zV[r,rt,Role,Clinic] <= 1
 0 <= zV[c,rt,Role,Clinic] <= 1
 0 <= zV[r,lb,Role,Clinic] <= 1
 0 <= zV[c,lb,Role,Clinic] <= 1
 0 <= zV[r,rt,Role,Location] <= 1
 0 <= zV[c,rt,Role,Location] <= 1
 0 <= zV[r,lb,Role,Location] <= 1
 0 <= zV[c,lb,Role,Location] <= 1
 0 <= zV[r,rt,Role,Therapist] <= 1
 0 <= zV[c,rt,Role,Therapist] <= 1
 0 <= zV[r,lb,Role,Therapist] <= 1
 0 <= zV[c,lb,Role,Therapist] <= 1
 0 <= zV[r,rt,Role,Instructor] <= 1
 0 <= zV[c,rt,Role,Instructor] <= 1
 0 <= zV[r,lb,Role,Instructor] <= 1
 0 <= zV[c,lb,Role,Instructor] <= 1
 0 <= zV[r,rt,User,CourseSession] <= 1
 0 <= zV[c,rt,User,CourseSession] <= 1
 0 <= zV[r,lb,User,CourseSession] <= 1
 0 <= zV[c,lb,User,CourseSession] <= 1
 0 <= zV[r,rt,User,Clinic] <= 1
 0 <= zV[c,rt,User,Clinic] <= 1
 0 <= zV[r,lb,User,Clinic] <= 1
 0 <= zV[c,lb,User,Clinic] <= 1
 0 <= zV[r,rt,User,Location] <= 1
 0 <= zV[c,rt,User,Location] <= 1
 0 <= zV[r,lb,User,Location] <= 1
 0 <= zV[c,lb,User,Location] <= 1
 0 <= zV[r,rt,User,Therapist] <= 1
 0 <= zV[c,rt,User,Therapist] <= 1
 0 <= zV[r,lb,User,Therapist] <= 1
 0 <= zV[c,lb,User,Therapist] <= 1
 0 <= zV[r,rt,User,Instructor] <= 1
 0 <= zV[c,rt,User,Instructor] <= 1
 0 <= zV[r,lb,User,Instructor] <= 1
 0 <= zV[c,lb,User,Instructor] <= 1
 0 <= zV[r,rt,CourseSession,Clinic] <= 1
 0 <= zV[c,rt,CourseSession,Clinic] <= 1
 0 <= zV[r,lb,CourseSession,Clinic] <= 1
 0 <= zV[c,lb,CourseSession,Clinic] <= 1
 0 <= zV[r,rt,CourseSession,Location] <= 1
 0 <= zV[c,rt,CourseSession,Location] <= 1
 0 <= zV[r,lb,CourseSession,Location] <= 1
 0 <= zV[c,lb,CourseSession,Location] <= 1
 0 <= zV[r,rt,CourseSession,Therapist] <= 1
 0 <= zV[c,rt,CourseSession,Therapist] <= 1
 0 <= zV[r,lb,CourseSession,Therapist] <= 1
 0 <= zV[c,lb,CourseSession,Therapist] <= 1
 0 <= zV[r,rt,CourseSession,Instructor] <= 1
 0 <= zV[c,rt,CourseSession,Instructor] <= 1
 0 <= zV[r,lb,CourseSession,Instructor] <= 1
 0 <= zV[c,lb,CourseSession,Instructor] <= 1
 0 <= zV[r,rt,Clinic,Location] <= 1
 0 <= zV[c,rt,Clinic,Location] <= 1
 0 <= zV[c,lb,Therapist,Instructor] <= 1
 1 <= x[c,Therapist] <= 32
 1 <= x[c,Location] <= 32
 1 <= x[c,Clinic] <= 32
 1 <= x[c,CourseSession] <= 29
 1 <= x[c,User] <= 32
 1 <= x[c,Role] <= 33
 1 <= x[c,Course] <= 33
 1 <= x[c,CertificateTemplate] <= 31
 1 <= x[c,Client] <= 32
 1 <= x[r,Instructor] <= 33
 1 <= x[r,Therapist] <= 33
 1 <= x[r,Location] <= 34
 1 <= x[r,Clinic] <= 35
 1 <= x[r,CourseSession] <= 34
 1 <= x[r,User] <= 36
 1 <= x[r,Role] <= 37
 1 <= x[r,Course] <= 37
 1 <= x[r,CertificateTemplate] <= 36
 1 <= x[r,Client] <= 34
 1 <= x[c,Instructor] <= 31
 center_dist[c,Therapist] free
 center_dist[c,Instructor] free
 0 <= dist[r,Client,CertificateTemplate] <= 43
 0 <= dist[r,Client,CourseSession] <= 42
 0 <= dist[r,Client,Instructor] <= 42.5
 0 <= dist[r,Client,Location] <= 42
 0 <= dist[r,Client,User] <= 43
 0 <= dist[r,CertificateTemplate,CourseSession] <= 43
 0 <= dist[r,Course,CourseSession] <= 43.5
 0 <= dist[r,CourseSession,Instructor] <= 42.5
 0 <= dist[r,Instructor,Location] <= 42.5
 0 <= dist[r,Clinic,Location] <= 42.5
 0 <= dist[r,Clinic,Therapist] <= 43
 0 <= dist[r,Location,Therapist] <= 42.5
 0 <= dist[r,Location,CourseSession] <= 42
 0 <= dist[r,Therapist,CourseSession] <= 42.5
 0 <= dist[r,Role,User] <= 42.5
 0 <= dist[r,User,Instructor] <= 43.5
 0 <= dist[r,User,Therapist] <= 43.5
 0 <= dist[c,Client,CertificateTemplate] <= 42.5
 0 <= dist[c,Client,CourseSession] <= 43.5
 0 <= dist[c,Client,Instructor] <= 42.5
 0 <= dist[c,Client,Location] <= 42
 0 <= dist[c,Client,User] <= 42
 0 <= dist[c,CertificateTemplate,CourseSession] <= 43
 0 <= dist[c,Course,CourseSession] <= 44
 0 <= dist[c,CourseSession,Instructor] <= 43
 0 <= dist[c,Instructor,Location] <= 42.5
 0 <= dist[c,Clinic,Location] <= 42
 0 <= dist[c,Clinic,Therapist] <= 42
 0 <= dist[c,Location,Therapist] <= 42
 0 <= dist[c,Location,CourseSession] <= 43.5
 0 <= dist[c,Therapist,CourseSession] <= 43.5
 0 <= dist[c,Role,User] <= 42.5
 0 <= dist[c,User,Instructor] <= 42.5
 0 <= dist[c,User,Therapist] <= 42
 center_dist[r,Client] free
 center_dist[r,CertificateTemplate] free
 center_dist[r,Course] free
 center_dist[r,Role] free
 center_dist[r,User] free
 center_dist[r,CourseSession] free
 center_dist[r,Clinic] free
 center_dist[r,Location] free
 center_dist[r,Therapist] free
 center_dist[r,Instructor] free
 center_dist[c,Client] free
 center_dist[c,CertificateTemplate] free
 center_dist[c,Course] free
 center_dist[c,Role] free
 center_dist[c,User] free
 center_dist[c,CourseSession] free
 center_dist[c,Clinic] free
 center_dist[c,Location] free
Binaries
 zV[r,lb,Therapist,Instructor] zV[c,rt,Therapist,Instructor] zV[r,rt,Therapist,Instructor] zV[c,lb,Location,Instructor]
 zV[r,lb,Location,Instructor] zV[c,rt,Location,Instructor] zV[r,rt,Location,Instructor] zV[c,lb,Location,Therapist]
 zV[r,lb,Location,Therapist] zV[c,rt,Location,Therapist] zV[r,rt,Location,Therapist] zV[c,lb,Clinic,Instructor]
 zV[r,lb,Clinic,Instructor] zV[c,rt,Clinic,Instructor] zV[r,rt,Clinic,Instructor] zV[c,lb,Clinic,Therapist]
 zV[r,lb,Clinic,Therapist] zV[c,rt,Clinic,Therapist] zV[r,rt,Clinic,Therapist] zV[c,lb,Clinic,Location]
 zV[r,lb,Clinic,Location] zV[r,rt,Client,CertificateTemplate] zV[c,rt,Client,CertificateTemplate] zV[r,lb,Client,CertificateTemplate]
 zV[c,lb,Client,CertificateTemplate] zV[r,rt,Client,Course] zV[c,rt,Client,Course] zV[r,lb,Client,Course]
 zV[c,lb,Client,Course] zV[r,rt,Client,Role] zV[c,rt,Client,Role] zV[r,lb,Client,Role] zV[c,lb,Client,Role]
 zV[r,rt,Client,User] zV[c,rt,Client,User] zV[r,lb,Client,User] zV[c,lb,Client,User] zV[r,rt,Client,CourseSession]
 zV[c,rt,Client,CourseSession] zV[r,lb,Client,CourseSession] zV[c,lb,Client,CourseSession] zV[r,rt,Client,Clinic]
 zV[c,rt,Client,Clinic] zV[r,lb,Client,Clinic] zV[c,lb,Client,Clinic] zV[r,rt,Client,Location] zV[c,rt,Client,Location]
 zV[r,lb,Client,Location] zV[c,lb,Client,Location] zV[r,rt,Client,Therapist] zV[c,rt,Client,Therapist]
 zV[r,lb,Client,Therapist] zV[c,lb,Client,Therapist] zV[r,rt,Client,Instructor] zV[c,rt,Client,Instructor]
 zV[r,lb,Client,Instructor] zV[c,lb,Client,Instructor] zV[r,rt,CertificateTemplate,Course] zV[c,rt,CertificateTemplate,Course]
 zV[r,lb,CertificateTemplate,Course] zV[c,lb,CertificateTemplate,Course] zV[r,rt,CertificateTemplate,Role]
 zV[c,rt,CertificateTemplate,Role] zV[r,lb,CertificateTemplate,Role] zV[c,lb,CertificateTemplate,Role]
 zV[r,rt,CertificateTemplate,User] zV[c,rt,CertificateTemplate,User] zV[r,lb,CertificateTemplate,User]
 zV[c,lb,CertificateTemplate,User] zV[r,rt,CertificateTemplate,CourseSession] zV[c,rt,CertificateTemplate,CourseSession]
 zV[r,lb,CertificateTemplate,CourseSession] zV[c,lb,CertificateTemplate,CourseSession] zV[r,rt,CertificateTemplate,Clinic]
 zV[c,rt,CertificateTemplate,Clinic] zV[r,lb,CertificateTemplate,Clinic] zV[c,lb,CertificateTemplate,Clinic]
 zV[r,rt,CertificateTemplate,Location] zV[c,rt,CertificateTemplate,Location] zV[r,lb,CertificateTemplate,Location]
 zV[c,lb,CertificateTemplate,Location] zV[r,rt,CertificateTemplate,Therapist] zV[c,rt,CertificateTemplate,Therapist]
 zV[r,lb,CertificateTemplate,Therapist] zV[c,lb,CertificateTemplate,Therapist] zV[r,rt,CertificateTemplate,Instructor]
 zV[c,rt,CertificateTemplate,Instructor] zV[r,lb,CertificateTemplate,Instructor] zV[c,lb,CertificateTemplate,Instructor]
 zV[r,rt,Course,Role] zV[c,rt,Course,Role] zV[r,lb,Course,Role] zV[c,lb,Course,Role] zV[r,rt,Course,User]
 zV[c,rt,Course,User] zV[r,lb,Course,User] zV[c,lb,Course,User] zV[r,rt,Course,CourseSession] zV[c,rt,Course,CourseSession]
 zV[r,lb,Course,CourseSession] zV[c,lb,Course,CourseSession] zV[r,rt,Course,Clinic] zV[c,rt,Course,Clinic]
 zV[r,lb,Course,Clinic] zV[c,lb,Course,Clinic] zV[r,rt,Course,Location] zV[c,rt,Course,Location] zV[r,lb,Course,Location]
 zV[c,lb,Course,Location] zV[r,rt,Course,Therapist] zV[c,rt,Course,Therapist] zV[r,lb,Course,Therapist]
 zV[c,lb,Course,Therapist] zV[r,rt,Course,Instructor] zV[c,rt,Course,Instructor] zV[r,lb,Course,Instructor]
 zV[c,lb,Course,Instructor] zV[r,rt,Role,User] zV[c,rt,Role,User] zV[r,lb,Role,User] zV[c,lb,Role,User]
 zV[r,rt,Role,CourseSession] zV[c,rt,Role,CourseSession] zV[r,lb,Role,CourseSession] zV[c,lb,Role,CourseSession]
 zV[r,rt,Role,Clinic] zV[c,rt,Role,Clinic] zV[r,lb,Role,Clinic] zV[c,lb,Role,Clinic] zV[r,rt,Role,Location]
 zV[c,rt,Role,Location] zV[r,lb,Role,Location] zV[c,lb,Role,Location] zV[r,rt,Role,Therapist] zV[c,rt,Role,Therapist]
 zV[r,lb,Role,Therapist] zV[c,lb,Role,Therapist] zV[r,rt,Role,Instructor] zV[c,rt,Role,Instructor] zV[r,lb,Role,Instructor]
 zV[c,lb,Role,Instructor] zV[r,rt,User,CourseSession] zV[c,rt,User,CourseSession] zV[r,lb,User,CourseSession]
 zV[c,lb,User,CourseSession] zV[r,rt,User,Clinic] zV[c,rt,User,Clinic] zV[r,lb,User,Clinic] zV[c,lb,User,Clinic]
 zV[r,rt,User,Location] zV[c,rt,User,Location] zV[r,lb,User,Location] zV[c,lb,User,Location] zV[r,rt,User,Therapist]
 zV[c,rt,User,Therapist] zV[r,lb,User,Therapist] zV[c,lb,User,Therapist] zV[r,rt,User,Instructor] zV[c,rt,User,Instructor]
 zV[r,lb,User,Instructor] zV[c,lb,User,Instructor] zV[r,rt,CourseSession,Clinic] zV[c,rt,CourseSession,Clinic]
 zV[r,lb,CourseSession,Clinic] zV[c,lb,CourseSession,Clinic] zV[r,rt,CourseSession,Location] zV[c,rt,CourseSession,Location]
 zV[r,lb,CourseSession,Location] zV[c,lb,CourseSession,Location] zV[r,rt,CourseSession,Therapist] zV[c,rt,CourseSession,Therapist]
 zV[r,lb,CourseSession,Therapist] zV[c,lb,CourseSession,Therapist] zV[r,rt,CourseSession,Instructor] zV[c,rt,CourseSession,Instructor]
 zV[r,lb,CourseSession,Instructor] zV[c,lb,CourseSession,Instructor] zV[r,rt,Clinic,Location] zV[c,rt,Clinic,Location]
 zV[c,lb,Therapist,Instructor]
Generals
 x[c,Therapist] x[c,Location] x[c,Clinic] x[c,CourseSession] x[c,User] x[c,Role] x[c,Course] x[c,CertificateTemplate]
 x[c,Client] x[r,Instructor] x[r,Therapist] x[r,Location] x[r,Clinic] x[r,CourseSession] x[r,User] x[r,Role]
 x[r,Course] x[r,CertificateTemplate] x[r,Client] x[c,Instructor]
End
