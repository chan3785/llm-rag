import type { AnswerSource } from "@/lib/chat";

const fmt = (n?: number) => (n === undefined ? "-" : n.toFixed(2));

function describe(source: AnswerSource): string {
  switch (source.reason) {
    case "matched":
      return `${source.landmark} · 유사도 ${fmt(source.score)} (기준 ${fmt(source.threshold)}) · 비슷한 사진 ${source.total}장 중 ${source.votes}장`;
    case "below_threshold":
      return `일치하는 명소 없음 · 가장 가까운 후보 ${source.candidate} ${fmt(source.score)} (기준 ${fmt(source.threshold)})`;
    case "no_reference_data":
      return "등록된 명소 데이터 없음";
    case "retrieval_failed":
      return "명소 검색 실패";
    case "no_image":
      return "이미지 없음";
  }
}

// Shows which stage decided the answer: RAG (landmark search ran, matched or not) or the VLM
// alone (no search was possible).
export function AnswerSourceBadge({ source }: { source: AnswerSource }) {
  const isRag = source.stage === "rag";
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      <span
        className={`rounded-full px-2 py-0.5 font-semibold ${
          isRag
            ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-200"
            : "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200"
        }`}
      >
        {isRag ? "RAG 판단" : "VLM 판단"}
      </span>
      <span className="text-gray-600 dark:text-gray-400">{describe(source)}</span>
    </div>
  );
}
