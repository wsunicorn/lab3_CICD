# =============================================================================
# Jenkinsfile - CI/CD Pipeline cho ML
# Đây là Declarative Pipeline syntax của Jenkins
# 
# Pipeline gồm 5 stages:
#   1. Checkout    -> Lấy code từ Git
#   2. Install     -> Cài Python dependencies
#   3. Train       -> Huấn luyện model
#   4. Test        -> Chạy test suite (quality gate)
#   5. Deploy      -> Deploy nếu test pass
#
# Cách dùng: Tạo Jenkins Job, chọn "Pipeline", 
# chọn "Pipeline script from SCM", chỉ vào repo Git
# =============================================================================

pipeline {
    agent any  // Chạy trên bất kỳ agent nào available

    // Biến môi trường dùng trong pipeline
    environment {
        PYTHON_VERSION = "3.10"
        MODEL_VERSION  = "1.0.${BUILD_NUMBER}"   // Auto-increment theo build number
        APP_NAME       = "ml-fraud-detector"
        // Credentials được lưu trong Jenkins Credentials Store
        // DOCKER_REGISTRY = credentials('docker-registry-credentials')
    }

    // Trigger: Tự động chạy khi có commit mới
    triggers {
        pollSCM("H/5 * * * *")  // Poll Git mỗi 5 phút
        // Hoặc dùng webhook: githubPush()
    }

    stages {
        // --------------------------------------------------------
        // STAGE 1: CHECKOUT SOURCE CODE
        // --------------------------------------------------------
        stage("Checkout") {
            steps {
                echo "Checking out source code..."
                checkout scm  // Lấy code từ SCM đã cấu hình
                
                // In thông tin commit để tracking
                sh "git log --oneline -5"
            }
        }

        // --------------------------------------------------------
        // STAGE 2: CÀI ĐẶT DEPENDENCIES
        // --------------------------------------------------------
        stage("Install Dependencies") {
            steps {
                echo "Installing Python dependencies..."
                sh """
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                """
            }
        }

        // --------------------------------------------------------
        // STAGE 3: CHẠY LINT (kiểm tra code style)
        // --------------------------------------------------------
        stage("Lint") {
            steps {
                echo "Running code linting..."
                sh """
                    . venv/bin/activate
                    pip install flake8
                    flake8 train.py test_model.py deploy.py --max-line-length=120 || true
                """
                // || true: không fail nếu có lint warning (tùy policy)
            }
        }

        // --------------------------------------------------------
        // STAGE 4: TRAIN MODEL
        // --------------------------------------------------------
        stage("Train Model") {
            steps {
                echo "Training ML model..."
                sh """
                    . venv/bin/activate
                    python train.py
                """
            }
            post {
                success {
                    echo "Training hoàn thành!"
                    // Archive model artifacts để có thể download từ Jenkins
                    archiveArtifacts artifacts: "models/**,metrics/**", fingerprint: true
                }
            }
        }

        // --------------------------------------------------------
        // STAGE 5: TEST & QUALITY GATE
        // Pipeline fail ở đây nếu model không đủ chất lượng
        // --------------------------------------------------------
        stage("Test Model") {
            steps {
                echo "Running model tests (quality gate)..."
                sh """
                    . venv/bin/activate
                    pytest test_model.py -v --tb=short \
                        --junitxml=test-results/results.xml \
                        --cov=train --cov-report=xml
                """
            }
            post {
                always {
                    // Publish test results vào Jenkins UI
                    junit "test-results/results.xml"
                }
                failure {
                    echo "Test FAILED - Pipeline sẽ dừng, KHÔNG deploy!"
                    // Có thể gửi notification: Slack, email,...
                    // slackSend channel: '#ml-alerts', message: "Model tests failed on ${env.BUILD_URL}"
                }
            }
        }

        // --------------------------------------------------------
        // STAGE 6: DEPLOY (chỉ chạy nếu test pass)
        // --------------------------------------------------------
        stage("Deploy to Staging") {
            when {
                // Chỉ deploy khi push vào branch main/master
                branch "main"
                // và tất cả stages trước phải success
            }
            steps {
                echo "Deploying model to staging..."
                sh """
                    . venv/bin/activate
                    python deploy.py deploy
                """
            }
        }

        // --------------------------------------------------------
        // STAGE 7: SMOKE TEST sau khi deploy
        // --------------------------------------------------------
        stage("Smoke Test") {
            when {
                branch "main"
            }
            steps {
                echo "Running smoke test on deployed model..."
                sh """
                    . venv/bin/activate
                    python -c "
import joblib, numpy as np
model = joblib.load('deployments/\$(cat deployments/current.json | python -c \"import sys,json; print(json.load(sys.stdin)[\\\"current_deploy\\\"])\")/model.pkl')
pred = model.predict(np.zeros((1, 8)))
print('Smoke test passed! Prediction:', pred)
"
                """
            }
        }
    }

    // --------------------------------------------------------
    // POST: Hành động sau khi pipeline kết thúc
    // --------------------------------------------------------
    post {
        success {
            echo "Pipeline PASSED! Model deployed successfully."
        }
        failure {
            echo "Pipeline FAILED!"
            // Tự động rollback nếu deploy thất bại
            sh """
                . venv/bin/activate
                python deploy.py rollback || true
            """
        }
        always {
            // Cleanup workspace
            cleanWs(cleanWhenNotBuilt: false, deleteDirs: true, disableDeferredWipeout: true)
        }
    }
}
