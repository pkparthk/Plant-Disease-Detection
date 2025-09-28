import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation } from "@tanstack/react-query";
import { Upload, AlertCircle, CheckCircle, Loader2 } from "lucide-react";
import { predict } from "@/services/api";
import { PredictionResult } from "@/types/api";
import { PredictionCard } from "@/components/PredictionCard";

export function HomePage() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const mutation = useMutation({
    mutationFn: predict,
    onSuccess: (data) => {
      setPrediction(data);
    },
    onError: (error) => {
      console.error("Prediction error:", error);
    },
  });

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        setPrediction(null);
        mutation.mutate(file);
      }
    },
    [mutation]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <div className="container py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Plant Disease Detection
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload an image of a plant leaf to detect diseases and get treatment
            recommendations using our AI-powered system.
          </p>
        </div>

        {/* Upload Area */}
        <div className="mb-8">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
              isDragActive
                ? "border-primary-500 bg-primary-50"
                : "border-gray-300 hover:border-primary-400 hover:bg-gray-50"
            }`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              <Upload className="h-12 w-12 text-gray-400" />
              <div>
                <p className="text-lg font-medium text-gray-900">
                  {isDragActive
                    ? "Drop the image here..."
                    : "Drag and drop an image, or click to select"}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  PNG, JPG, JPEG, WEBP up to 10MB
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {mutation.isPending && (
          <div className="card p-6 mb-8">
            <div className="flex items-center justify-center space-x-3">
              <Loader2 className="h-6 w-6 animate-spin text-primary-600" />
              <span className="text-lg font-medium text-gray-900">
                Analyzing image...
              </span>
            </div>
          </div>
        )}

        {/* Error State */}
        {mutation.isError && (
          <div className="card p-6 mb-8 border-danger-200 bg-danger-50">
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-6 w-6 text-danger-600" />
              <div>
                <h3 className="text-lg font-medium text-danger-900">
                  Analysis Failed
                </h3>
                <p className="text-danger-700 mt-1">
                  {(mutation.error as any)?.message ||
                    "An error occurred while analyzing the image."}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Success State */}
        {prediction && !mutation.isPending && (
          <div className="space-y-6">
            <div className="card p-6 border-primary-200 bg-primary-50">
              <div className="flex items-center space-x-3">
                <CheckCircle className="h-6 w-6 text-primary-600" />
                <div>
                  <h3 className="text-lg font-medium text-primary-900">
                    Analysis Complete
                  </h3>
                  <p className="text-primary-700 mt-1">
                    Disease detection analysis has been completed successfully.
                  </p>
                </div>
              </div>
            </div>

            <PredictionCard prediction={prediction} />
          </div>
        )}

        {/* Features */}
        {!prediction && !mutation.isPending && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <Upload className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Easy Upload
              </h3>
              <p className="text-gray-600">
                Simply drag and drop or click to upload plant images
              </p>
            </div>

            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <AlertCircle className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                AI Detection
              </h3>
              <p className="text-gray-600">
                Advanced machine learning models detect diseases accurately
              </p>
            </div>

            <div className="text-center">
              <div className="bg-primary-100 rounded-full p-3 w-12 h-12 mx-auto mb-4 flex items-center justify-center">
                <CheckCircle className="h-6 w-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                Treatment Guide
              </h3>
              <p className="text-gray-600">
                Get detailed treatment recommendations for identified diseases
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
