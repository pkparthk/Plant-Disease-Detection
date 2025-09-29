import {
  PredictionResult,
  ModelInfo,
  HealthStatus,
  ApiError,
} from "@/types/api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Debug logging
console.log("Environment variables:", import.meta.env);
console.log("VITE_API_URL:", import.meta.env.VITE_API_URL);
console.log("API_BASE_URL:", API_BASE_URL);

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {        
    this.baseUrl = baseUrl || "http://localhost:8000";    
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorData: ApiError = await response.json().catch(() => ({
          detail: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new Error(errorData.detail || "Request failed");
      }

      return response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error("Network error occurred");
    }
  }

  async predict(file: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append("image", file);

    // Force the correct URL
    const baseUrl = "http://localhost:8000";
    const url = `${baseUrl}/api/predict`;

    console.log("Making prediction request to:", url);
    console.log("Base URL:", baseUrl);
    console.log("this.baseUrl:", this.baseUrl);

    const response = await fetch(url, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData: ApiError = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`,
      }));
      throw new Error(errorData.detail || "Prediction failed");
    }

    const responseData = await response.json();
    console.log("API Response:", responseData);

    // Extract the top prediction from the response
    if (responseData.top_prediction) {
      // Add timestamp to the prediction result
      return {
        ...responseData.top_prediction,
        timestamp: responseData.timestamp,
      };
    }

    // Fallback: return the response as-is if structure is different
    return responseData;
  }

  async getHealth(): Promise<HealthStatus> {
    return this.request<HealthStatus>("/api/health");
  }

  async getModelInfo(): Promise<ModelInfo> {
    return this.request<ModelInfo>("/api/model/info");
  }

  async getDiseaseClasses(): Promise<string[]> {
    return this.request<string[]>("/api/model/classes");
  }
}

export const apiClient = new ApiClient();

// Export individual functions for easier use
export const { predict, getHealth, getModelInfo, getDiseaseClasses } =
  apiClient;
