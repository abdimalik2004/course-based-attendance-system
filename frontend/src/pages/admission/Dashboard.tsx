import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { CheckCircle2, Clock3, ScanFace, Users } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { useAdmissionStore } from "@/store/useAdmissionStore";

const statCards = [
  {
    key: "totalStudents",
    label: "Total Students",
    icon: Users,
    tone: "from-sky-500/20 to-cyan-500/10 text-sky-600 dark:text-sky-300",
    route: "/admission/students",
  },
  {
    key: "newAdmissions",
    label: "New Admissions",
    icon: CheckCircle2,
    tone: "from-emerald-500/20 to-green-500/10 text-emerald-600 dark:text-emerald-300",
    route: "/admission/students",
  },
  {
    key: "pendingApplications",
    label: "Pending Applications",
    icon: Clock3,
    tone: "from-amber-500/20 to-orange-500/10 text-amber-600 dark:text-amber-300",
    route: "/admission/students?status=Pending",
  },
  {
    key: "rejectedApplications",
    label: "Rejected Applications",
    icon: ScanFace,
    tone: "from-rose-500/20 to-red-500/10 text-rose-600 dark:text-rose-300",
    route: "/admission/students?status=Rejected",
  },
] as const;

export default function AdmissionDashboard() {
  const { dashboardStats, isLoading, error, fetchAdmissionData } =
    useAdmissionStore();
  const navigate = useNavigate();

  useEffect(() => {
    void fetchAdmissionData();
  }, [fetchAdmissionData]);

  return (
    <div className="space-y-6">
      <div className="space-y-2 border-b border-white/10 pb-5">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
          Admission Dashboard
        </h1>
        <p className="text-base text-gray-500 dark:text-gray-400">
          Overview of student admissions and registered enrollments.
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/10 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {statCards.map(({ key, label, icon: Icon, tone, route }) => (
          <Card
            key={key}
            className="overflow-hidden border-white/60 dark:border-white/10 cursor-pointer hover:-translate-y-1 hover:shadow-lg transition-all duration-300 active:scale-[0.98]"
            onClick={() => navigate(route)}
          >
            <CardContent className="p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {label}
                  </p>
                  <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                    {isLoading ? "--" : dashboardStats[key]}
                  </p>
                </div>
                <div className={`rounded-2xl bg-gradient-to-br p-3 ${tone}`}>
                  <Icon size={22} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_0.9fr]">
        <Card className="border-white/60 dark:border-white/10">
          <CardContent className="p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Quick actions
                </h2>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Jump into the most common admission tasks.
                </p>
              </div>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              <Link
                to="/admission/students"
                className="rounded-2xl border border-gray-200 bg-gray-50/70 p-4 transition hover:border-primary/30 hover:bg-primary/5 dark:border-white/10 dark:bg-white/5"
              >
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  Review students
                </p>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Search, edit, and maintain student records.
                </p>
              </Link>

              <Link
                to="/admission/approval"
                className="rounded-2xl border border-gray-200 bg-gray-50/70 p-4 transition hover:border-primary/30 hover:bg-primary/5 dark:border-white/10 dark:bg-white/5"
              >
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  Approve applications
                </p>
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  Work through the pending queue and captured photos.
                </p>
              </Link>
            </div>
          </CardContent>
        </Card>

        <Card className="border-white/60 dark:border-white/10">
          <CardContent className="p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Admission flow
            </h2>
            <div className="mt-5 space-y-4">
              {[
                "Capture a new student application.",
                "Review the uploaded face data.",
                "Approve or reject the applicant.",
              ].map((item, index) => (
                <div key={item} className="flex gap-3">
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    {index + 1}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {item}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
