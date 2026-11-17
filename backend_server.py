"""
AI 기반 개인 맞춤형 백색소음 추천 시스템 - FastAPI 백엔드 API 서버
파일명: backend_server.py
실행 명령: uvicorn backend_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
from recommendation_model import PersonalizedSoundRecommender, SOUND_IDS, SOUND_NAMES_KO

app = FastAPI(
    title="AI 기반 개인 맞춤형 백색소음 추천 시스템 API",
    description="사용자의 청각적 취향과 피드백 데이터를 분석하여 개인화된 백색소음 조합을 추천하는 인공지능 백엔드",
    version="1.0.0"
)

# CORS 설정 (React/Next.js/HTML 브라우저 연동 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인메모리 유저별 AI 모델 인스턴스 저장소 (실제 운영 시 DB와 연동)
user_models: Dict[str, PersonalizedSoundRecommender] = {}
user_saved_presets: Dict[str, List[Dict[str, Any]]] = {}

def get_or_create_recommender(user_id: str) -> PersonalizedSoundRecommender:
    if user_id not in user_models:
        user_models[user_id] = PersonalizedSoundRecommender(user_id)
    return user_models[user_id]


# --- Pydantic 데이터 모델 정의 ---
class SoundInfo(BaseModel):
    id: str
    name_ko: str
    category: str
    description: str

class FeedbackRequest(BaseModel):
    user_id: str = "student_user"
    satisfaction_score: int = Field(..., ge=1, le=5, description="만족도 1~5점")
    active_ratios: Dict[str, float] = Field(..., description="각 사운드별 믹싱 비율 (%)")
    master_volume: int = Field(60, ge=0, le=100)
    played_duration_seconds: int = Field(60, description="실제 청취 시간(초)")
    volume_feedback: Optional[str] = "optimal"       # too_loud, optimal, too_quiet
    bass_feedback: Optional[str] = "optimal"         # too_strong, optimal, too_weak
    treble_feedback: Optional[str] = "optimal"       # too_sharp, optimal, too_dull
    is_monotonous: Optional[bool] = False            # 단조로움 여부
    desired_addition: Optional[str] = None          # 추가 희망 사운드
    desired_removal: Optional[str] = None           # 제거 희망 사운드
    desired_mood: Optional[str] = "focus"           # 희망 분위기
    detailed_feedback: Optional[str] = ""            # 자유 주관식 의견

class PresetSaveRequest(BaseModel):
    user_id: str = "student_user"
    preset_name: str
    ratios: Dict[str, float]
    eq: Dict[str, float]
    master_volume: int


# --- API 엔드포인트 정의 ---

@app.get("/")
def read_root():
    return {
        "project": "AI 기반 개인 맞춤형 백색소음 추천 시스템",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/api/sounds", response_model=List[SoundInfo])
def get_sounds():
    """7가지 사운드 목록 및 정보 반환"""
    categories = {
        "brown_noise": ("Noise", "저주파 대역이 풍부하여 집중 및 차폐에 탁월한 소리"),
        "pink_noise": ("Noise", "1/f 뇌파 동조 스펙트럼으로 수면과 힐링에 좋은 소리"),
        "white_noise": ("Noise", "전 주파수가 균일하여 외부 말소리를 마스킹하는 소리"),
        "nature_wind": ("Nature", "부드러운 나뭇잎 스침과 바람의 자연스러운 흐름"),
        "water_river": ("Nature", "지속적인 청량감과 긴장 완화를 주는 시냇물/강물 소리"),
        "bird_rain": ("Nature", "감성을 자극하고 스트레스를 풀어주는 빗소리와 숲속 새소리"),
        "campfire": ("Ambient", "따뜻하고 아늑한 공간감을 선사하는 타닥타닥 모닥불 소리")
    }
    return [
        SoundInfo(
            id=sid,
            name_ko=SOUND_NAMES_KO[sid],
            category=categories[sid][0],
            description=categories[sid][1]
        )
        for sid in SOUND_IDS
    ]

@app.get("/api/recommendations")
def get_recommendations(user_id: str = "student_user"):
    """사용자의 취향을 학습한 AI 개인 맞춤형 백색소음 추천 3종 반환"""
    recommender = get_or_create_recommender(user_id)
    recs = recommender.generate_recommendations()
    return {
        "user_id": user_id,
        "recommendations": recs,
        "learned_iterations": len(recommender.history)
    }

@app.post("/api/feedback")
def submit_feedback(data: FeedbackRequest):
    """
    사용자의 만족도 평가 데이터를 수집하고 AI 모델을 즉시 재학습
    """
    recommender = get_or_create_recommender(data.user_id)
    
    session_config = {
        "ratios": data.active_ratios,
        "duration": data.played_duration_seconds,
        "master_volume": data.master_volume
    }
    
    feedback_details = {
        "volume_feedback": data.volume_feedback,
        "bass_feedback": data.bass_feedback,
        "treble_feedback": data.treble_feedback,
        "is_monotonous": data.is_monotonous,
        "desired_addition": data.desired_addition,
        "desired_removal": data.desired_removal,
        "desired_mood": data.desired_mood,
        "detailed_feedback": data.detailed_feedback
    }
    
    result = recommender.update_with_feedback(
        session_config=session_config,
        satisfaction_score=data.satisfaction_score,
        feedback_details=feedback_details
    )
    
    return {
        "message": "피드백이 성공적으로 반영되어 AI 추천 모델이 갱신되었습니다.",
        "result": result
    }

@app.get("/api/user-profile")
def get_user_profile(user_id: str = "student_user"):
    """사용자의 청각 취향 벡터, 누적 청취 통계, 레이더 차트용 데이터 제공"""
    recommender = get_or_create_recommender(user_id)
    
    total_duration = sum(h["session_config"].get("duration", 0) for h in recommender.history)
    scores = [h["score"] for h in recommender.history]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "user_id": user_id,
        "sound_weights": recommender.sound_weights,
        "preferred_bass": round(recommender.preferred_bass, 1),
        "preferred_treble": round(recommender.preferred_treble, 1),
        "preferred_smoothness": round(recommender.preferred_smoothness, 2),
        "evaluation_count": len(recommender.history),
        "total_listening_minutes": round(total_duration / 60.0, 1),
        "average_satisfaction": avg_score,
        "history": recommender.history[-10:] # 최근 10건
    }

@app.post("/api/presets")
def save_preset(preset: PresetSaveRequest):
    """사용자 맞춤 프리셋 저장"""
    if preset.user_id not in user_saved_presets:
        user_saved_presets[preset.user_id] = []
    
    new_preset = {
        "name": preset.preset_name,
        "ratios": preset.ratios,
        "eq": preset.eq,
        "master_volume": preset.master_volume
    }
    user_saved_presets[preset.user_id].append(new_preset)
    return {"message": "프리셋이 저장되었습니다.", "preset": new_preset}

@app.get("/api/presets")
def get_presets(user_id: str = "student_user"):
    """사용자가 저장한 프리셋 목록 조회"""
    return user_saved_presets.get(user_id, [])


if __name__ == "__main__":
    uvicorn.run("backend_server:app", host="127.0.0.1", port=8000, reload=True)

