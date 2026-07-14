import { useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Search, Filter, CheckCircle2, XCircle, X, Copy, Check, AlertTriangle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Modal } from "@/components/ui/Modal";
import { Badge } from "@/components/ui/Badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { useAdmissionStore } from "@/store/useAdmissionStore";
import type { Student } from "@/store/useAdmissionStore";
import { ViewButton } from "@/components/ui/ViewButton";
import admissionService, {
  type StudentCapturedImageDto,
} from "@/services/admissionService";

type StudentCapturedImagePreview = StudentCapturedImageDto & {
  previewUrl: string;
};

/** Map raw DTO → Student shape (same as useAdmissionStore normalizeAdmissionState) */
function mapToStudent(dto: any): Student {
  return {
    id: String(dto.id),
    studentNumber: dto.student_number,
    fullName: dto.full_name,
    facultyId: String(dto.faculty_id),
    departmentId: String(dto.department_id),
    faculty: dto.faculty_name ?? `Faculty ${dto.faculty_id}`,
    department: dto.department_name ?? `Department ${dto.department_id}`,
    class: `Class ${new Date(dto.created_at).getFullYear()}`,
    faceImagesCount: dto.face_images_count ?? 0,
    status: dto.status,
    createdAt: dto.created_at,
    dateOfBirth: dto.date_of_birth ?? null,
    phone: dto.phone ?? null,
    email: dto.email ?? null,
  };
}

export default function StudentsApproval() {
  const { isSaving, approveStudent, rejectStudent } = useAdmissionStore();
  const queryClient = useQueryClient();
  const [searchTerm, setSearchTerm] = useState("");

  // Fetch ALL pending students directly from the API — bypasses the store's
  // single-page cache so students on page 2+ are never invisible in the queue.
  const {
    data: pendingData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["admissionPendingStudents"],
    queryFn: () =>
      admissionService.listStudents({ status: "pending", skip: 0, limit: 500 }),
    staleTime: 30_000,
  });

  // Modal states
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);

  const [isApproveConfirmOpen, setIsApproveConfirmOpen] = useState(false);
  const [isRejectConfirmOpen, setIsRejectConfirmOpen] = useState(false);
  const [approvalError, setApprovalError] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<{ studentNumber: string; password: string } | null>(null);
  const [copiedField, setCopiedField] = useState<"username" | "password" | null>(null);
  const [capturedImages, setCapturedImages] = useState<
    StudentCapturedImagePreview[]
  >([]);
  const [isImagesLoading, setIsImagesLoading] = useState(false);
  const [imagesError, setImagesError] = useState<string | null>(null);

  const allPendingStudents: Student[] = useMemo(() => {
    const items = pendingData?.items ?? [];
    return items.map(mapToStudent);
  }, [pendingData]);

  const pendingStudents = useMemo(() => {
    if (!searchTerm) return allPendingStudents;
    const q = searchTerm.toLowerCase();
    return allPendingStudents.filter(
      (s) =>
        s.fullName.toLowerCase().includes(q) ||
        s.studentNumber.toLowerCase().includes(q),
    );
  }, [allPendingStudents, searchTerm]);

  const openViewModal = (student: Student) => {
    setSelectedStudent(student);
    setIsViewModalOpen(true);
  };

  useEffect(() => {
    let isActive = true;
    const createdObjectUrls: string[] = [];

    const fetchImages = async () => {
      if (!isViewModalOpen || !selectedStudent) {
        setCapturedImages([]);
        setImagesError(null);
        return;
      }

      setIsImagesLoading(true);
      setImagesError(null);
      try {
        const data = await admissionService.getStudentCapturedImages(
          Number(selectedStudent.id),
        );

        const previewImages = await Promise.all(
          (data.images ?? []).map(async (image) => {
            if (
              /^https?:\/\//i.test(image.url) ||
              image.url.startsWith("blob:") ||
              image.url.startsWith("data:")
            ) {
              return {
                ...image,
                previewUrl: image.url,
              };
            }

            const blob = await admissionService.getStudentCapturedImageBlob(
              Number(selectedStudent.id),
              image.file_name,
            );
            const objectUrl = URL.createObjectURL(blob);
            createdObjectUrls.push(objectUrl);

            return {
              ...image,
              previewUrl: objectUrl,
            };
          }),
        );

        if (isActive) {
          setCapturedImages(previewImages);
        }
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load captured images.";
        setImagesError(message);
        setCapturedImages([]);
      } finally {
        if (isActive) {
          setIsImagesLoading(false);
        }
      }
    };

    void fetchImages();

    return () => {
      isActive = false;
      createdObjectUrls.forEach((objectUrl) => URL.revokeObjectURL(objectUrl));
    };
  }, [isViewModalOpen, selectedStudent]);

  const closeModals = () => {
    setIsViewModalOpen(false);
    setSelectedStudent(null);
    setCapturedImages([]);
    setImagesError(null);
  };

  const handleApproveClick = () => {
    setIsApproveConfirmOpen(true);
  };

  const handleRejectClick = () => {
    setIsRejectConfirmOpen(true);
  };

  const copyToClipboard = (text: string, field: "username" | "password") => {
    void navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const confirmApprove = async () => {
    if (!selectedStudent) return;
    try {
      const result = await approveStudent(selectedStudent.id);
      setIsApproveConfirmOpen(false);
      closeModals();
      // Refresh the pending list so the approved student disappears from the queue
      void queryClient.invalidateQueries({ queryKey: ["admissionPendingStudents"] });
      if (result?.generatedPassword) {
        setCredentials({ studentNumber: result.studentNumber, password: result.generatedPassword });
      }
    } catch (err: unknown) {
      setIsApproveConfirmOpen(false);
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setApprovalError(
        detail ?? "Failed to approve student. Please try again.",
      );
    }
  };

  const confirmReject = async () => {
    if (selectedStudent) {
      await rejectStudent(selectedStudent.id);
      setIsRejectConfirmOpen(false);
      closeModals();
      // Refresh the pending list so the rejected student disappears from the queue
      void queryClient.invalidateQueries({ queryKey: ["admissionPendingStudents"] });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
            Students Approval
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Review newly registered pending students.
          </p>
        </div>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/10 dark:text-red-300">
          {error instanceof Error ? error.message : "Failed to load pending students."}
        </div>
      ) : null}

      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
                <Search size={18} />
              </div>
              <Input
                type="text"
                placeholder="Search by name or student number..."
                className="pl-10"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <Button variant="secondary" className="shrink-0">
              <Filter size={18} className="mr-2" />
              Filter
            </Button>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ST-NO</TableHead>
                <TableHead>ST-Name</TableHead>
                <TableHead>Faculty</TableHead>
                <TableHead>Department</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <TableRow key={`pending-skeleton-${i}`}>
                    <TableCell>
                      <div className="h-4 w-20 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-32 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-24 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-24 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-6 w-20 rounded-full bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="ml-auto h-8 w-16 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                    </TableCell>
                  </TableRow>
                ))
              ) : pendingStudents.length > 0 ? (
                pendingStudents.map((student) => (
                  <TableRow key={student.id}>
                    <TableCell className="font-medium text-primary">
                      {student.studentNumber}
                    </TableCell>
                    <TableCell className="font-medium text-gray-900 dark:text-white">
                      {student.fullName}
                    </TableCell>
                    <TableCell>{student.faculty}</TableCell>
                    <TableCell>{student.department}</TableCell>
                    <TableCell>
                      <Badge variant="warning">
                        {student.status.charAt(0).toUpperCase() +
                          student.status.slice(1)}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <ViewButton
                          onClick={() => openViewModal(student)}
                          tooltip="Review"
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="text-center py-8 text-gray-500"
                  >
                    No pending students found for approval.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* View & Approval Modal */}
      <Modal
        isOpen={isViewModalOpen}
        onClose={closeModals}
        title="Student Details Review"
        className="max-w-4xl"
      >
        {selectedStudent && (
          <div className="space-y-8">
            {/* Student Information Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  ST-NO
                </p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {selectedStudent.studentNumber}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Full Name
                </p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {selectedStudent.fullName}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Faculty
                </p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {selectedStudent.faculty}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Department
                </p>
                <p className="font-medium text-gray-900 dark:text-white">
                  {selectedStudent.department}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Status
                </p>
                <Badge variant="warning" className="mt-1">
                  Pending
                </Badge>
              </div>
            </div>

            {/* Images Captured Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center justify-between">
                <span>Captured Images</span>
                <span className="text-sm font-normal text-gray-500 bg-gray-100 dark:bg-white/10 px-3 py-1 rounded-full">
                  {capturedImages.length} Images
                </span>
              </h3>

              {isImagesLoading ? (
                <div className="grid grid-cols-4 sm:grid-cols-5 gap-3 max-h-[400px] overflow-y-auto custom-scrollbar pr-2 pb-2">
                  {Array.from({ length: 10 }).map((_, idx) => (
                    <div
                      key={`img-skeleton-${idx}`}
                      className="aspect-square rounded-xl border border-gray-200 dark:border-white/10 bg-gray-200 dark:bg-white/10 animate-pulse"
                    />
                  ))}
                </div>
              ) : imagesError ? (
                <div className="flex flex-col items-center justify-center p-8 bg-red-50 dark:bg-red-500/10 rounded-xl border border-red-100 dark:border-red-500/20">
                  <p className="text-red-600 dark:text-red-300 text-sm">
                    {imagesError}
                  </p>
                </div>
              ) : capturedImages.length > 0 ? (
                <div className="grid grid-cols-4 sm:grid-cols-5 gap-3 max-h-[400px] overflow-y-auto custom-scrollbar pr-2 pb-2">
                  {capturedImages.map((image) => (
                    <div
                      key={image.file_name}
                      className="relative aspect-square rounded-xl overflow-hidden border border-gray-200 dark:border-white/10 group bg-gray-100 dark:bg-white/5"
                    >
                      <img
                        src={image.previewUrl}
                        alt={image.file_name}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                        loading="lazy"
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 bg-gray-50 dark:bg-white/5 rounded-xl border border-gray-100 dark:border-white/10">
                  <p className="text-gray-500 dark:text-gray-400">
                    No captured images found in the dataset for this student.
                  </p>
                </div>
              )}
            </div>

            {/* No images warning — shown when images are loaded but none exist */}
            {!isImagesLoading && !imagesError && capturedImages.length === 0 && (
              <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20">
                <AlertTriangle className="text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" size={18} />
                <div>
                  <p className="text-sm font-semibold text-amber-800 dark:text-amber-300">
                    No face images captured
                  </p>
                  <p className="text-sm text-amber-700 dark:text-amber-400 mt-0.5">
                    This student has no face data registered. Approval will be blocked until face images are captured via the Face Registration page.
                  </p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="pt-4 border-t border-gray-200 dark:border-white/10 flex flex-wrap items-center justify-end gap-3">
              <Button
                variant="secondary"
                onClick={closeModals}
                className="min-w-[100px]"
              >
                <X size={18} className="mr-2" />
                Cancel
              </Button>
              <Button
                onClick={handleRejectClick}
                disabled={isSaving}
                className="min-w-[120px] bg-red-500 hover:bg-red-600 text-white border-transparent"
              >
                <XCircle size={18} className="mr-2" />
                Reject
              </Button>
              <Button
                onClick={handleApproveClick}
                disabled={isSaving}
                className="min-w-[120px] bg-emerald-500 hover:bg-emerald-600 text-white border-transparent shadow-lg shadow-emerald-500/20"
              >
                <CheckCircle2 size={18} className="mr-2" />
                Approve
              </Button>
            </div>
          </div>
        )}
      </Modal>

      {/* Approve Confirm Modal */}
      <Modal
        isOpen={isApproveConfirmOpen}
        onClose={() => setIsApproveConfirmOpen(false)}
        title="Confirm Approval"
      >
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            Are you sure to approve{" "}
            <span className="font-bold text-gray-900 dark:text-white">
              {selectedStudent?.fullName}
            </span>{" "}
            Admission?
          </p>
          <div className="flex justify-end gap-3 pt-2">
            <Button
              variant="secondary"
              onClick={() => setIsApproveConfirmOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-emerald-500 hover:bg-emerald-600 text-white border-transparent shadow-lg shadow-emerald-500/20"
              disabled={isSaving}
              onClick={confirmApprove}
            >
              Approve
            </Button>
          </div>
        </div>
      </Modal>

      {/* Reject Confirm Modal */}
      <Modal
        isOpen={isRejectConfirmOpen}
        onClose={() => setIsRejectConfirmOpen(false)}
        title="Confirm Rejection"
      >
        <div className="space-y-6">
          <p className="text-gray-700 dark:text-gray-300">
            Are you sure to Reject{" "}
            <span className="font-bold text-gray-900 dark:text-white">
              {selectedStudent?.fullName}
            </span>{" "}
            Admission?
          </p>
          <div className="flex justify-end gap-3 pt-2">
            <Button
              variant="secondary"
              onClick={() => setIsRejectConfirmOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-red-500 hover:bg-red-600 text-white border-transparent shadow-lg shadow-red-500/20"
              disabled={isSaving}
              onClick={confirmReject}
            >
              Reject
            </Button>
          </div>
        </div>
      </Modal>

      {/* Approval Error Modal — shown when backend rejects the approval (e.g. no face images) */}
      <Modal
        isOpen={!!approvalError}
        onClose={() => setApprovalError(null)}
        title="Cannot Approve Student"
      >
        <div className="space-y-5">
          <div className="flex items-start gap-4 p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-500/20 flex items-center justify-center shrink-0">
              <AlertTriangle className="text-red-600 dark:text-red-400" size={20} />
            </div>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              {approvalError}
            </p>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Go to the <span className="font-medium text-gray-700 dark:text-gray-300">Face Registration</span> page, capture the student's face images, then return here to approve.
          </p>
          <div className="flex justify-end">
            <Button onClick={() => setApprovalError(null)}>Got It</Button>
          </div>
        </div>
      </Modal>

      {/* Credentials Modal — shown after approval */}
      <Modal
        isOpen={!!credentials}
        onClose={() => setCredentials(null)}
        title="Student Login Credentials"
      >
        {credentials && (
          <div className="space-y-6">
            <div className="rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20 p-4">
              <p className="text-sm text-emerald-700 dark:text-emerald-300 font-medium">
                ✓ Admission approved. Hand these credentials to the student so they can log in.
              </p>
            </div>

            {/* Username */}
            <div className="space-y-1">
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Username</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 rounded-lg bg-gray-100 dark:bg-white/10 border border-gray-200 dark:border-white/10 px-4 py-3 font-mono text-base font-semibold text-gray-900 dark:text-white">
                  {credentials.studentNumber}
                </div>
                <button
                  onClick={() => copyToClipboard(credentials.studentNumber, "username")}
                  className="p-2.5 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
                >
                  {copiedField === "username" ? <Check size={16} className="text-emerald-500" /> : <Copy size={16} />}
                </button>
              </div>
            </div>

            {/* Password */}
            <div className="space-y-1">
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Password</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 rounded-lg bg-gray-100 dark:bg-white/10 border border-gray-200 dark:border-white/10 px-4 py-3 font-mono text-base font-semibold text-gray-900 dark:text-white">
                  {credentials.password}
                </div>
                <button
                  onClick={() => copyToClipboard(credentials.password, "password")}
                  className="p-2.5 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
                >
                  {copiedField === "password" ? <Check size={16} className="text-emerald-500" /> : <Copy size={16} />}
                </button>
              </div>
            </div>

            <p className="text-xs text-gray-400 dark:text-gray-500">
              The student can change their password after logging in for the first time.
            </p>

            <div className="flex justify-end pt-2">
              <Button
                className="bg-emerald-500 hover:bg-emerald-600 text-white border-transparent"
                onClick={() => setCredentials(null)}
              >
                Done
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
