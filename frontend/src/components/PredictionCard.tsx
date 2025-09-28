import { PredictionResult } from "@/types/api";
import {
  AlertTriangle,
  Clock,
  Shield,
  Zap,
  Eye,
  AlertCircle,
  DollarSign,
  Activity,
} from "lucide-react";

interface PredictionCardProps {
  prediction: PredictionResult;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "low":
        return "text-green-600 bg-green-100";
      case "medium":
        return "text-yellow-600 bg-yellow-100";
      case "high":
        return "text-red-600 bg-red-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "low":
        return <Shield className="h-4 w-4" />;
      case "medium":
        return <Clock className="h-4 w-4" />;
      case "high":
        return <AlertTriangle className="h-4 w-4" />;
      default:
        return <Zap className="h-4 w-4" />;
    }
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="card p-6">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">
            Detection Results
          </h2>
          <div className="text-sm text-gray-500">
            {prediction.timestamp
              ? formatDate(prediction.timestamp)
              : new Date().toLocaleString()}
          </div>
        </div>

        {/* Disease Information */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Disease</h3>
            <p className="text-lg font-semibold text-gray-900">
              {prediction.disease === "None"
                ? `Healthy ${prediction.plant}`
                : prediction.disease}
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Confidence
            </h3>
            <p className="text-lg font-semibold text-gray-900">
              {formatConfidence(prediction.confidence)}
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Severity</h3>
            <div
              className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(
                prediction.severity
              )}`}
            >
              {getSeverityIcon(prediction.severity)}
              <span className="capitalize">{prediction.severity}</span>
            </div>
          </div>
        </div>

        {/* Disease Details */}
        {(prediction.symptoms?.length ||
          prediction.causes ||
          prediction.urgency ||
          prediction.economic_impact) && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Disease Information
            </h3>

            {/* Symptoms */}
            {prediction.symptoms && prediction.symptoms.length > 0 && (
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2 flex items-center">
                  <Eye className="h-4 w-4 mr-2" />
                  Symptoms to Look For
                </h4>
                <ul className="text-blue-800 text-sm space-y-1">
                  {prediction.symptoms.map((symptom, index) => (
                    <li key={index} className="flex items-start">
                      <span className="mr-2">•</span>
                      <span>{symptom}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Causes */}
            {prediction.causes && prediction.causes.trim().length > 0 && (
              <div className="bg-amber-50 rounded-lg p-4">
                <h4 className="font-medium text-amber-900 mb-2 flex items-center">
                  <AlertCircle className="h-4 w-4 mr-2" />
                  What Causes This Disease
                </h4>
                <p className="text-amber-800 text-sm">{prediction.causes}</p>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Urgency */}
              {prediction.urgency && prediction.urgency.trim().length > 0 && (
                <div className="bg-red-50 rounded-lg p-4">
                  <h4 className="font-medium text-red-900 mb-2 flex items-center">
                    <Activity className="h-4 w-4 mr-2" />
                    Treatment Urgency
                  </h4>
                  <p className="text-red-800 text-sm">{prediction.urgency}</p>
                </div>
              )}

              {/* Economic Impact */}
              {prediction.economic_impact &&
                prediction.economic_impact.trim().length > 0 && (
                  <div className="bg-purple-50 rounded-lg p-4">
                    <h4 className="font-medium text-purple-900 mb-2 flex items-center">
                      <DollarSign className="h-4 w-4 mr-2" />
                      Economic Impact
                    </h4>
                    <p className="text-purple-800 text-sm">
                      {prediction.economic_impact}
                    </p>
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Treatment Recommendations */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Treatment Recommendations
          </h3>

          {/* Chemical Treatments */}
          {prediction.treatment?.chemical &&
            prediction.treatment.chemical.trim().length > 0 && (
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2 flex items-center">
                  <Zap className="h-4 w-4 mr-2" />
                  Chemical Treatments
                </h4>
                <p className="text-blue-800 text-sm">
                  {prediction.treatment.chemical}
                </p>
              </div>
            )}

          {/* Cultural Treatments */}
          {prediction.treatment?.cultural &&
            prediction.treatment.cultural.trim().length > 0 && (
              <div className="bg-green-50 rounded-lg p-4">
                <h4 className="font-medium text-green-900 mb-2 flex items-center">
                  <Shield className="h-4 w-4 mr-2" />
                  Cultural Practices
                </h4>
                <p className="text-green-800 text-sm">
                  {prediction.treatment.cultural}
                </p>
              </div>
            )}

          {/* Preventive Measures */}
          {prediction.treatment?.preventive &&
            prediction.treatment.preventive.trim().length > 0 && (
              <div className="bg-yellow-50 rounded-lg p-4">
                <h4 className="font-medium text-yellow-900 mb-2 flex items-center">
                  <Clock className="h-4 w-4 mr-2" />
                  Preventive Measures
                </h4>
                <p className="text-yellow-800 text-sm">
                  {prediction.treatment.preventive}
                </p>
              </div>
            )}
        </div>

        {/* Footer Note */}
        <div className="bg-gray-50 rounded-lg p-4">
          <p className="text-sm text-gray-600">
            <strong>Note:</strong> These recommendations are generated by AI
            analysis. For severe cases or professional farming operations,
            please consult with agricultural experts or local extension services
            for personalized advice.
          </p>
        </div>
      </div>
    </div>
  );
}
