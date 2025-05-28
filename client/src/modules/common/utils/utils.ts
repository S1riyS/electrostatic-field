import { SerializedError } from "@reduxjs/toolkit";
import { FetchBaseQueryError } from "@reduxjs/toolkit/query";

export function isStrictFloat(str: string): boolean {
  return /^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/.test(str.trim());
}

export function formatApiError(
  error: FetchBaseQueryError | SerializedError | undefined,
): string {
  if (!error) return "Произошла неизвестная ошибка.";

  if ("status" in error) {
    // Network / fetch failure
    if (error.status === "FETCH_ERROR") {
      return "Не удалось подключиться к серверу. Проверьте соединение.";
    }

    // Server responded with HTTP status and possibly our typed ApiError
    if (typeof error.status === "number") {
      const data = error.data as { message?: string; statusCode?: number };

      if (data?.message) {
        return data.message;
      }

      return `Ошибка ${error.status}`;
    }
  }

  // Fallback for unexpected serialized errors
  if ("message" in error && typeof error.message === "string") {
    return error.message;
  }

  return "Произошла неизвестная ошибка.";
}
