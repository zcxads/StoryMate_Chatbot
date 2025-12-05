# 🚀 StoryMate 챗봇 API CI/CD 자동 배포 가이드

StoryMate 챗봇 API를 GitHub Actions를 통해 자동화된 CI/CD 파이프라인으로 배포하는 방법을 안내합니다.

## 📋 CI/CD 파이프라인 개요

### 자동화된 배포 과정
- **코드 푸시**: main 브랜치에 코드 푸시
- **자동 테스트**: Python 테스트 및 린팅 실행
- **자동 빌드**: Docker 이미지 빌드
- **자동 배포**: Docker Hub에 이미지 푸시

### 장점
- 코드 변경 시 자동 배포
- 테스트 통과 후에만 배포
- 일관된 배포 과정
- 배포 과정 추적 가능

## 🔄 CI/CD 자동 배포

### GitHub Actions 워크플로우

프로젝트에는 `.github/workflows/deploy.yml` 파일이 설정되어 있어 main 브랜치에 푸시할 때마다 자동으로 배포됩니다.

#### 워크플로우 과정:

1. **테스트 단계**
   - Python 3.11 환경 설정
   - 의존성 설치
   - pytest 테스트 실행

2. **빌드 및 배포 단계**
   - Docker 이미지 빌드
   - Docker Hub에 이미지 푸시
   - 최신 태그와 커밋 SHA 태그 생성

### 필요한 GitHub Secrets 설정

#### Docker Hub 인증 정보:
```bash
DOCKER_USERNAME=your_docker_username
DOCKER_PASSWORD=your_docker_password
```

#### API 키들:
```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

#### GitHub Secrets 설정 방법:

1. **GitHub 저장소 설정**
   - 저장소 페이지 → Settings → Secrets and variables → Actions
   - "New repository secret" 클릭

2. **필요한 Secrets 추가**
   ```
   DOCKER_USERNAME: Docker Hub 사용자명
   DOCKER_PASSWORD: Docker Hub 비밀번호 또는 액세스 토큰
   OPENAI_API_KEY: OpenAI API 키
   GEMINI_API_KEY: Google Gemini API 키
   ```

## 🐳 자동 배포된 컨테이너 실행

### 1. 최신 이미지 가져오기
```bash
docker pull your-username/storymate-api:latest
```

### 2. Docker Compose로 실행 (권장)

#### 서비스 실행
```bash
# 백그라운드에서 서비스 시작
docker compose up -d

# 로그 확인
docker compose logs -f api
```

#### 기존 컨테이너가 있는 경우 새 컨테이너 실행

##### 1. 기존 컨테이너 확인
```bash
# 현재 실행 중인 컨테이너 확인
docker compose ps

# 모든 컨테이너 확인 (중지된 것도)
docker compose ps -a

# 특정 이름의 컨테이너 확인
docker ps | grep storymate
```

##### 2. 기존 컨테이너 중지 및 제거
```bash
# Docker Compose로 실행된 컨테이너 중지
docker compose down

# 볼륨과 네트워크까지 모두 제거
docker compose down --volumes --remove-orphans
```

##### 3. 새 컨테이너 실행
```bash
# 최신 이미지 가져오기
docker pull your-username/storymate-api:latest

# 새 컨테이너 실행
docker compose up -d

# 실행 확인
docker compose ps
```

## 📊 배포 모니터링

### 배포된 서비스 모니터링

#### 컨테이너 상태 확인:
```bash
# 실행 중인 컨테이너 확인
docker ps

# 컨테이너 로그 확인
docker logs storymate-api
```
