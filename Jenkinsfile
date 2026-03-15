pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "alokprakash1587/wine_predict_2022bcs0014"
    }

    stages {

        stage('Checkout') {
            steps {
                git credentialsId: 'git-creds', url: 'https://github.com/speedracer2145/MLOpslab2.git'
            }
        }

        stage('Setup Python') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    ACCURACY = sh(
                        script: "jq '.accuracy' app/artifacts/metrics.json",
                        returnStdout: true
                    ).trim()
                    echo "Accuracy: ${ACCURACY}"
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $DOCKER_IMAGE .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'USER',
                    passwordVariable: 'PASS'
                )]) {
                    sh '''
                    echo $PASS | docker login -u $USER --password-stdin
                    docker push $DOCKER_IMAGE
                    '''
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**'
        }
    }
}
