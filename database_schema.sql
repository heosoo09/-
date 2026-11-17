-- =====================================================================
-- AI 기반 개인 맞춤형 백색소음 추천 시스템 - 데이터베이스 스키마
-- Database: PostgreSQL / SQLite 호환 스키마 설계
-- =====================================================================

-- 1. 사용자 테이블 (Users)
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    preferred_environment VARCHAR(50) DEFAULT 'study', -- study, sleep, relax, focus
    data_consent BOOLEAN DEFAULT TRUE,               -- AI 학습 데이터 수집 동의 여부
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 소리 원본 메타데이터 테이블 (Sound Sources)
CREATE TABLE IF NOT EXISTS sound_sources (
    sound_id VARCHAR(50) PRIMARY KEY,
    name_ko VARCHAR(100) NOT NULL,                  -- 예: 브라운 사운드, 모닥불 소리
    name_en VARCHAR(100) NOT NULL,                  -- 예: Brown Noise, Fireplace
    category VARCHAR(50) NOT NULL,                  -- Noise(잡음), Nature(자연), Ambient(환경)
    base_frequency_type VARCHAR(20) NOT NULL,       -- low(저주파), mid(중주파), high(고주파), balanced(균형)
    spectral_tilt_db_per_oct FLOAT DEFAULT 0.0,     -- 주파수 기울기 (-6dB: Brown, -3dB: Pink, 0dB: White)
    masking_power_score FLOAT DEFAULT 0.5,          -- 소음 차폐 능력 점수 (0.0 ~ 1.0)
    calmness_score FLOAT DEFAULT 0.5,               -- 심리적 이완도 점수 (0.0 ~ 1.0)
    focus_score FLOAT DEFAULT 0.5,                  -- 집중도 유도 점수 (0.0 ~ 1.0)
    default_volume INT DEFAULT 50,                  -- 기본 권장 볼륨 (0 ~ 100)
    is_synthesizable BOOLEAN DEFAULT TRUE           -- 알고리즘 합성 지원 여부
);

-- 3. 소리별 특성 벡터 데이터 등록 (초기 7대 사운드)
INSERT INTO sound_sources (sound_id, name_ko, name_en, category, base_frequency_type, spectral_tilt_db_per_oct, masking_power_score, calmness_score, focus_score, default_volume, is_synthesizable)
VALUES 
('brown_noise', '브라운 사운드', 'Brown Noise', 'Noise', 'low', -6.0, 0.95, 0.85, 0.90, 60, TRUE),
('pink_noise', '핑크 사운드', 'Pink Noise', 'Noise', 'mid', -3.0, 0.90, 0.80, 0.85, 50, TRUE),
('white_noise', '화이트 사운드', 'White Noise', 'Noise', 'high', 0.0, 0.85, 0.60, 0.95, 40, TRUE),
('nature_wind', '자연(바람) 소리', 'Nature Sounds', 'Nature', 'low', -4.5, 0.70, 0.90, 0.75, 45, TRUE),
('water_river', '물(강물) 소리', 'Water Sounds', 'Nature', 'mid', -2.5, 0.80, 0.95, 0.80, 50, TRUE),
('bird_rain', '새소리/빗소리', 'Bird/Rain Sounds', 'Nature', 'high', -1.5, 0.75, 0.90, 0.70, 40, TRUE),
('campfire', '모닥불 소리', 'Fireplace Sounds', 'Ambient', 'balanced', -3.5, 0.65, 0.95, 0.65, 45, TRUE)
ON CONFLICT (sound_id) DO NOTHING;

-- 4. 사용자 취향 프로필 테이블 (User Preference Profiles)
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(50) PRIMARY KEY,
    preferred_bass_level FLOAT DEFAULT 0.0,         -- 저음 선호도 (-10.0 ~ +10.0 dB)
    preferred_treble_level FLOAT DEFAULT 0.0,       -- 고음 선호도 (-10.0 ~ +10.0 dB)
    preferred_smoothness FLOAT DEFAULT 0.5,         -- 소리 부드러움 필터 선호 (0.0 ~ 1.0)
    preferred_master_volume INT DEFAULT 65,         -- 선호 마스터 볼륨 (0 ~ 100)
    sound_weights_json TEXT,                        -- 7가지 소리별 가중치 (JSON)
    learning_iteration_count INT DEFAULT 0,         -- 피드백 반영 학습 횟수
    last_trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 5. 재생 세션 및 조절 이력 테이블 (Playback Sessions)
CREATE TABLE IF NOT EXISTS playback_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    active_sounds_json TEXT NOT NULL,               -- 선택된 소리 목록 및 볼륨 비율
    master_volume INT NOT NULL,                     -- 전체 음량
    bass_eq FLOAT DEFAULT 0.0,                      -- 저음 설정값
    treble_eq FLOAT DEFAULT 0.0,                    -- 고음 설정값
    smoothness FLOAT DEFAULT 0.5,                   -- 소리 부드러움 설정값
    played_duration_seconds INT DEFAULT 0,          -- 실제 재생 시간 (초 단위)
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    settings_changed_count INT DEFAULT 0,           -- 재생 도중 사용자가 조절한 횟수
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 6. 사용자 만족도 평가 데이터 테이블 (Satisfaction Evaluations)
CREATE TABLE IF NOT EXISTS satisfaction_evaluations (
    evaluation_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    satisfaction_score INT NOT NULL CHECK (satisfaction_score BETWEEN 1 AND 5), -- 만족도 (1~5점)
    volume_feedback VARCHAR(20),                    -- 'too_loud', 'optimal', 'too_quiet'
    bass_feedback VARCHAR(20),                      -- 'too_strong', 'optimal', 'too_weak'
    treble_feedback VARCHAR(20),                    -- 'too_sharp', 'optimal', 'too_dull'
    is_monotonous BOOLEAN DEFAULT FALSE,            -- 소리가 너무 단조로운지 여부
    desired_addition VARCHAR(50),                   -- 추가하고 싶은 사운드 ID
    desired_removal VARCHAR(50),                    -- 제거하고 싶은 사운드 ID
    desired_mood VARCHAR(50),                       -- 희망 분위기 (집중, 수면, 힐링 등)
    detailed_feedback TEXT,                         -- 사용자 직접 입력 개선 의견
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES playback_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 7. AI 추천 결과 및 이력 테이블 (Recommendation History)
CREATE TABLE IF NOT EXISTS recommendation_history (
    recommendation_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) DEFAULT 'v1.2-hybrid',
    recommended_preset_json TEXT NOT NULL,          -- 추천된 소리 조합 및 비율 (예: Brown 50%, Water 30%, Fire 20%)
    recommendation_reason TEXT NOT NULL,            -- AI의 추천 근거 설명
    confidence_score FLOAT,                         -- 모델 신뢰도 점수 (0.0 ~ 1.0)
    user_reaction VARCHAR(20),                      -- 'applied', 'liked', 'disliked', 'ignored'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 8. 저장된 사용자 프리셋 (Saved Presets)
CREATE TABLE IF NOT EXISTS user_presets (
    preset_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    preset_name VARCHAR(100) NOT NULL,
    preset_config_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 9. 빠른 조회를 위한 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_sessions_user ON playback_sessions(user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_evals_user ON satisfaction_evaluations(user_id, satisfaction_score);
CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendation_history(user_id, created_at);

