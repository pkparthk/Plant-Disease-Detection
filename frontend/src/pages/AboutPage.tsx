import { Leaf, Users, Target, Zap } from "lucide-react";

export function AboutPage() {
  return (
    <div className="container py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            About Plant Disease Detection
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            An AI-powered system designed to help farmers, gardeners, and
            agricultural professionals identify plant diseases quickly and
            accurately.
          </p>
        </div>

        {/* Mission */}
        <div className="card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
          <p className="text-gray-700 text-lg leading-relaxed">
            We aim to democratize access to plant disease diagnosis using
            cutting-edge artificial intelligence. By making advanced
            agricultural expertise available to everyone, we help protect crops,
            increase yields, and promote sustainable farming practices
            worldwide.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">Key Features</h2>

            <div className="flex items-start space-x-4">
              <div className="bg-primary-100 rounded-full p-2 flex-shrink-0">
                <Leaf className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">
                  25+ Disease Classes
                </h3>
                <p className="text-gray-600">
                  Detects diseases across multiple plant species including
                  apple, corn, grape, potato, and tomato.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-primary-100 rounded-full p-2 flex-shrink-0">
                <Zap className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">
                  Real-time Analysis
                </h3>
                <p className="text-gray-600">
                  Get instant disease detection results with confidence scores
                  and severity assessments.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-primary-100 rounded-full p-2 flex-shrink-0">
                <Target className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">
                  Treatment Recommendations
                </h3>
                <p className="text-gray-600">
                  Receive detailed chemical, cultural, and preventive treatment
                  options for identified diseases.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-primary-100 rounded-full p-2 flex-shrink-0">
                <Users className="h-5 w-5 text-primary-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">
                  User-Friendly Interface
                </h3>
                <p className="text-gray-600">
                  Simple drag-and-drop interface designed for users of all
                  technical backgrounds.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">Technology</h2>

            <div className="card p-6 bg-gray-50">
              <h3 className="font-semibold text-gray-900 mb-3">
                Machine Learning Models
              </h3>
              <ul className="space-y-2 text-gray-700">
                <li>• TensorFlow deep learning models</li>
                <li>• PyTorch neural networks</li>
                <li>• ONNX model support</li>
                <li>• Convolutional Neural Networks (CNNs)</li>
              </ul>
            </div>

            <div className="card p-6 bg-gray-50">
              <h3 className="font-semibold text-gray-900 mb-3">
                Modern Web Stack
              </h3>
              <ul className="space-y-2 text-gray-700">
                <li>• React + TypeScript frontend</li>
                <li>• FastAPI Python backend</li>
                <li>• Docker containerization</li>
                <li>• Redis caching</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Supported Diseases */}
        <div className="card p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Supported Plant Diseases
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Apple</h3>
              <ul className="space-y-1 text-gray-700 text-sm">
                <li>• Apple Scab</li>
                <li>• Black Rot</li>
                <li>• Cedar Apple Rust</li>
                <li>• Healthy</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Corn</h3>
              <ul className="space-y-1 text-gray-700 text-sm">
                <li>• Cercospora Leaf Spot</li>
                <li>• Common Rust</li>
                <li>• Northern Leaf Blight</li>
                <li>• Healthy</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Grape</h3>
              <ul className="space-y-1 text-gray-700 text-sm">
                <li>• Black Rot</li>
                <li>• Esca (Black Measles)</li>
                <li>• Leaf Blight</li>
                <li>• Healthy</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Potato</h3>
              <ul className="space-y-1 text-gray-700 text-sm">
                <li>• Early Blight</li>
                <li>• Late Blight</li>
                <li>• Healthy</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Tomato</h3>
              <ul className="space-y-1 text-gray-700 text-sm">
                <li>• Bacterial Spot</li>
                <li>• Early Blight</li>
                <li>• Late Blight</li>
                <li>• Leaf Mold</li>
                <li>• Septoria Leaf Spot</li>
                <li>• Spider Mites</li>
                <li>• Target Spot</li>
                <li>• Yellow Leaf Curl Virus</li>
                <li>• Mosaic Virus</li>
                <li>• Healthy</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="card p-6 bg-yellow-50 border-yellow-200">
          <h3 className="font-semibold text-yellow-900 mb-2">
            Important Disclaimer
          </h3>
          <p className="text-yellow-800 text-sm">
            This AI system provides disease detection suggestions based on image
            analysis. Results should be used as a preliminary assessment only.
            For critical agricultural decisions, professional diagnosis from
            qualified agricultural experts or plant pathologists is recommended.
            The accuracy of predictions may vary based on image quality,
            lighting conditions, and disease progression stages.
          </p>
        </div>
      </div>
    </div>
  );
}
