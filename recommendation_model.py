"""
AI 기반 개인 맞춤형 백색소음 추천 시스템 - 추천 알고리즘 엔진
파일명: recommendation_model.py
기술 스택: Python, NumPy, Pandas, Scikit-learn 호환 벡터 연산

주요 특징:
1. 콘텐츠 기반 필터링 (Content-Based Filtering) 사운드 특성 매트릭스
2. 사용자 청각 취향 벡터 (User Acoustic Profile) 적응형 학습
3. 피드백 점수(1~5점) 및 불만족 요인 기반 가중치 재학습 (Adaptive Reinforcement)
4. 설명 가능한 AI (Explainable AI - XAI) 추천 사유 자동 생성 엔진
5. 콜드 스타트(초기 사용자 데이터 부족) 방지 기본 프리셋 제공
"""

import numpy as np
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime

# 7가지 사운드 ID 정의
SOUND_IDS = [
    "brown_noise",   # 브라운 사운드
    "pink_noise",    # 핑크 사운드
    "white_noise",   # 화이트 사운드
    "nature_wind",   # 자연소리(바람)
    "water_river",   # 물소리(강물)
    "bird_rain",     # 새소리/빗소리
    "campfire"       # 모닥불 소리
]

SOUND_NAMES_KO = {
    "brown_noise": "브라운 사운드",
    "pink_noise": "핑크 사운드",
    "white_noise": "화이트 사운드",
    "nature_wind": "자연(바람) 소리",
    "water_river": "물소리(강물)",
    "bird_rain": "새소리/빗소리",
    "campfire": "모닥불 소리"
}

# 사운드 메타데이터 및 음향 특성 벡터 (정규화된 특성값 0.0 ~ 1.0)
# 특성: [저주파 비중, 중고주파 비중, 소음 차폐력, 심리적 안정감, 집중력 유도, 역동성/리듬감]
SOUND_FEATURE_PROFILES = {
    "brown_noise": np.array([0.95, 0.10, 0.95, 0.85, 0.90, 0.15]),
    "pink_noise":  np.array([0.70, 0.40, 0.90, 0.80, 0.85, 0.20]),
    "white_noise": np.array([0.30, 0.90, 0.85, 0.60, 0.95, 0.10]),
    "nature_wind": np.array([0.65, 0.45, 0.70, 0.90, 0.75, 0.40]),
    "water_river": np.array([0.60, 0.55, 0.80, 0.95, 0.80, 0.50]),
    "bird_rain":   np.array([0.40, 0.85, 0.75, 0.90, 0.70, 0.60]),
    "campfire":    np.array([0.70, 0.60, 0.65, 0.95, 0.65, 0.75])
}


class PersonalizedSoundRecommender:
    """
    사용자의 청각적 취향과 피드백 데이터를 분석하여 
    최적의 백색소음 조합(사운드 종류, 비중 %, EQ)을 추천하는 AI 모델.
    """

    def __init__(self, user_id: str = "guest_user"):
        self.user_id = user_id
        # 기본 사운드 선호도 가중치 (균등 분포에서 출발)
        self.sound_weights: Dict[str, float] = {sid: 1.0 / len(SOUND_IDS) for sid in SOUND_IDS}
        # 선호 주파수 EQ (-10.0 ~ +10.0 dB)
        self.preferred_bass: float = 0.0
        self.preferred_treble: float = 0.0
        self.preferred_smoothness: float = 0.5
        # 누적 학습 이력
        self.history: List[Dict[str, Any]] = []
        self.learning_rate: float = 0.20

    def get_initial_templates(self) -> List[Dict[str, Any]]:
        """초기 사용자(데이터 부족 콜드스타트)를 위한 과학적 표준 조합 3종 제공"""
        return [
            {
                "id": "rec_focus_deep",
                "title": "깊은 몰입 집중형 (Deep Focus)",
                "description": "저주파 대역으로 외부 소음을 완벽히 차단하고 뇌파를 안정시키는 조합입니다.",
                "ratios": {"brown_noise": 50, "water_river": 30, "campfire": 20},
                "eq": {"bass": 2.5, "treble": -1.0, "smoothness": 0.6},
                "master_volume": 65,
                "reason": "소음 차폐력이 가장 높은 브라운 사운드를 베이스로 잔잔한 물소리와 모닥불의 크랙클이 집중력을 유지해 줍니다."
            },
            {
                "id": "rec_calm_healing",
                "title": "자연 힐링 & 스트레스 완화 (Calm Nature)",
                "description": "핑크 노이즈와 자연의 유기적인 사운드가 결합되어 심신 이완을 돕습니다.",
                "ratios": {"pink_noise": 40, "nature_wind": 35, "bird_rain": 25},
                "eq": {"bass": 0.0, "treble": 1.5, "smoothness": 0.7},
                "master_volume": 55,
                "reason": "1/f 주파수 특성을 가진 핑크 사운드와 자연 바람, 빗소리의 조화로 심리적 피로도를 낮추어 줍니다."
            },
            {
                "id": "rec_study_masking",
                "title": "고주파 마스킹 스터디 (Acoustic Masking)",
                "description": "도서관이나 카페 주변의 사람 목소리/웅성거림을 효과적으로 지워주는 조합입니다.",
                "ratios": {"white_noise": 50, "water_river": 35, "nature_wind": 15},
                "eq": {"bass": -1.0, "treble": 0.5, "smoothness": 0.5},
                "master_volume": 60,
                "reason": "화이트 노이즈의 넓은 스펙트럼과 일정한 유속의 물소리가 말소리 주파수를 완벽히 마스킹합니다."
            }
        ]

    def update_with_feedback(self, 
                             session_config: Dict[str, Any], 
                             satisfaction_score: int, 
                             feedback_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자의 만족도 평가(1~5점) 및 세부 피드백을 받아 개인화 가중치 재학습 (Reinforcement)
        
        피드백 처리 규칙:
        - 5점: 현재 조합을 매우 긍정적 지표로 반영 (큰 학습률 보정)
        - 4점: 현재 조합을 유지하며 소폭 미세 조정
        - 3점: 일부 사운드 비중 재조정
        - 2점: 불만족 사유(저음 과다, 고음 날카로움, 단조로움 등) 분석하여 반대 방향 보정
        - 1점: 탐색(Exploration) 가중치 대폭 증가, 대안 조합 생성
        """
        # 세션에서 활성화된 사운드 및 비율
        active_ratios = session_config.get("ratios", {})
        duration_seconds = session_config.get("duration", 60)
        # 재생 시간이 길수록 실제 청취 만족에 대한 신뢰도 가중치 증가 (로그 스케일)
        duration_weight = min(1.5, max(0.5, np.log1p(duration_seconds / 60.0)))

        # 1. 만족도 점수 정규화 (-1.0 ~ +1.0 스케일)
        # 1점: -1.0, 2점: -0.5, 3점: +0.1, 4점: +0.6, 5점: +1.0
        score_impact_map = {1: -1.0, 2: -0.5, 3: 0.1, 4: 0.6, 5: 1.0}
        impact = score_impact_map.get(satisfaction_score, 0.2) * duration_weight

        # 2. 사운드 가중치 업데이트
        total_active_pct = sum(active_ratios.values()) or 100
        for sid in SOUND_IDS:
            curr_ratio = active_ratios.get(sid, 0) / total_active_pct
            if satisfaction_score >= 4:
                # 긍정 피드백: 현재 사용 비율만큼 선호도 증가
                self.sound_weights[sid] += self.learning_rate * impact * curr_ratio
            elif satisfaction_score <= 2:
                # 부정 피드백: 현재 사용 비율만큼 감점하고, 사용 안 한 사운드 가중치 보존
                self.sound_weights[sid] -= self.learning_rate * abs(impact) * curr_ratio * 0.8
            else:
                # 3점 (보통): 미세 보정
                self.sound_weights[sid] += self.learning_rate * 0.05 * curr_ratio

            # 최소 가중치 보장 (다양성 유지를 위해 0.02 이하로 떨어지지 않음)
            self.sound_weights[sid] = max(0.02, self.sound_weights[sid])

        # 3. 세부 피드백에 따른 음향 EQ 보정
        # (1) 음량 피드백
        vol_fb = feedback_details.get("volume_feedback")
        if vol_fb == "too_loud":
            session_config["master_volume"] = max(20, session_config.get("master_volume", 60) - 10)
        elif vol_fb == "too_quiet":
            session_config["master_volume"] = min(100, session_config.get("master_volume", 60) + 10)

        # (2) 저음 피드백
        bass_fb = feedback_details.get("bass_feedback")
        if bass_fb == "too_strong":
            self.preferred_bass -= 1.5
        elif bass_fb == "too_weak":
            self.preferred_bass += 1.5

        # (3) 고음 피드백
        treble_fb = feedback_details.get("treble_feedback")
        if treble_fb == "too_sharp":
            self.preferred_treble -= 1.5
            self.preferred_smoothness = min(1.0, self.preferred_smoothness + 0.15)
        elif treble_fb == "too_dull":
            self.preferred_treble += 1.5

        # (4) 단조로움 피드백: 자연/앰비언트 계열 가중치 상향 (모닥불, 새소리, 물소리)
        if feedback_details.get("is_monotonous", False):
            self.sound_weights["campfire"] *= 1.4
            self.sound_weights["bird_rain"] *= 1.3
            self.sound_weights["water_river"] *= 1.2

        # (5) 추가/제거 희망 사운드
        desired_add = feedback_details.get("desired_addition")
        if desired_add and desired_add in self.sound_weights:
            self.sound_weights[desired_add] *= 1.5

        desired_rem = feedback_details.get("desired_removal")
        if desired_rem and desired_rem in self.sound_weights:
            self.sound_weights[desired_rem] *= 0.4

        # 가중치 재정규화 (합이 1이 되도록)
        weight_sum = sum(self.sound_weights.values())
        for sid in self.sound_weights:
            self.sound_weights[sid] /= weight_sum

        # 이력 기록
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "score": satisfaction_score,
            "feedback": feedback_details,
            "session_config": session_config,
            "learned_weights": dict(self.sound_weights)
        }
        self.history.append(history_entry)

        return {
            "status": "success",
            "updated_weights": self.sound_weights,
            "preferred_bass": self.preferred_bass,
            "preferred_treble": self.preferred_treble,
            "history_count": len(self.history)
        }

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        학습된 사용자 취향 프로필 및 콘텐츠 유사도를 바탕으로 
        3개의 맞춤형 추천 사운드 조합 생성
        """
        if len(self.history) == 0:
            return self.get_initial_templates()

        # 취향 상위 랭킹 사운드 추출
        sorted_sounds = sorted(self.sound_weights.items(), key=lambda item: item[1], reverse=True)
        top_sound_1, top_weight_1 = sorted_sounds[0]
        top_sound_2, top_weight_2 = sorted_sounds[1]
        top_sound_3, top_weight_3 = sorted_sounds[2]

        avg_score = np.mean([h["score"] for h in self.history])

        recommendations = []

        # 추천 1: [개인화 1위 기반 최적화 조합] 사용자가 가장 선호하는 사운드 중심 3-믹스
        ratio_1 = {
            top_sound_1: 50,
            top_sound_2: 30,
            top_sound_3: 20
        }
        reason_1 = (
            f"사용자 데이터 분석 결과 '{SOUND_NAMES_KO[top_sound_1]}'(선호도 {top_weight_1*100:.0f}%)와 "
            f"'{SOUND_NAMES_KO[top_sound_2]}'에 대한 만족도가 가장 높았으며, "
            f"저음 성향({self.preferred_bass:+.1f}dB)에 최적화하여 구성한 맞춤 조합입니다."
        )
        recommendations.append({
            "id": "rec_custom_top",
            "title": f"취향 맞춤 베스트 믹스 ({SOUND_NAMES_KO[top_sound_1]} 중심)",
            "description": "사용자의 최근 청취 및 만족도 이력이 가장 강력하게 반영된 1순위 추천입니다.",
            "ratios": ratio_1,
            "eq": {
                "bass": round(self.preferred_bass, 1),
                "treble": round(self.preferred_treble, 1),
                "smoothness": round(self.preferred_smoothness, 2)
            },
            "master_volume": 65,
            "confidence": min(0.98, 0.70 + 0.05 * len(self.history)),
            "reason": reason_1
        })

        # 추천 2: [조화형 밸런스 믹스] 1위 사운드 + 자연 힐링 요소 결합
        natural_candidates = ["water_river", "nature_wind", "bird_rain", "campfire"]
        # 1위가 자연 사운드가 아니면 자연 사운드 중 최고 선호도 사운드 배합
        best_natural = next((s for s, _ in sorted_sounds if s in natural_candidates), "water_river")
        best_noise = next((s for s, _ in sorted_sounds if s in ["brown_noise", "pink_noise", "white_noise"]), "brown_noise")
        ratio_2 = {
            best_noise: 45,
            best_natural: 35,
            "campfire" if best_natural != "campfire" else "nature_wind": 20
        }
        reason_2 = (
            f"주파수 안정성이 높은 {SOUND_NAMES_KO[best_noise]} 45%와 뇌 피로도를 완화하는 {SOUND_NAMES_KO[best_natural]} 35%를 조화롭게 믹싱하여 "
            f"단조로움을 줄이고 심리적 편안함을 극대화했습니다."
        )
        recommendations.append({
            "id": "rec_custom_harmony",
            "title": "안정 & 이완 밸런스 믹스",
            "description": "차폐 효과와 자연의 편안함을 균형 있게 배합한 조화형 조합입니다.",
            "ratios": ratio_2,
            "eq": {
                "bass": round(self.preferred_bass * 0.8, 1),
                "treble": round(self.preferred_treble * 0.8 + 0.5, 1),
                "smoothness": round(min(1.0, self.preferred_smoothness + 0.1), 2)
            },
            "master_volume": 60,
            "confidence": min(0.95, 0.65 + 0.05 * len(self.history)),
            "reason": reason_2
        })

        # 추천 3: [탐색형 대안 믹스 (Exploration)] 취향의 다양성을 넓히는 새로운 시도
        # 사용자가 덜 들었던 사운드(탐색 후보)
        least_used_sound = sorted_sounds[-1][0]
        ratio_3 = {
            top_sound_1: 40,
            sorted_sounds[2][0]: 35,
            least_used_sound: 25
        }
        reason_3 = (
            f"기존 선호 사운드인 {SOUND_NAMES_KO[top_sound_1]}의 베이스 위에, "
            f"새로운 청각적 리프레시를 위해 {SOUND_NAMES_KO[least_used_sound]}을 25% 조합하여 새로운 집중감을 제공합니다."
        )
        recommendations.append({
            "id": "rec_custom_explore",
            "title": "새로운 청각 자극 리프레시 믹스",
            "description": "익숙한 취향에 신선한 사운드 요소를 가미하여 집중력 저하를 방지합니다.",
            "ratios": ratio_3,
            "eq": {
                "bass": round(self.preferred_bass * 0.5, 1),
                "treble": round(self.preferred_treble * 0.5, 1),
                "smoothness": 0.5
            },
            "master_volume": 55,
            "confidence": 0.75,
            "reason": reason_3
        })

        return recommendations


# 간단 테스트 코드
if __name__ == "__main__":
    recommender = PersonalizedSoundRecommender("test_student")
    print("=== [1] 초기 콜드스타트 추천 템플릿 ===")
    initial_recs = recommender.get_initial_templates()
    for rec in initial_recs:
        print(f"[{rec['title']}] {rec['ratios']} -> {rec['reason']}")

    print("\n=== [2] 사용자 5점 만족도 피드백 반영 (브라운 사운드 50% + 물소리 30% + 모닥불 20%) ===")
    recommender.update_with_feedback(
        session_config={"ratios": {"brown_noise": 50, "water_river": 30, "campfire": 20}, "duration": 180, "master_volume": 65},
        satisfaction_score=5,
        feedback_details={"bass_feedback": "optimal", "treble_feedback": "optimal", "is_monotonous": False}
    )

    print("\n=== [3] 학습 후 개인 맞춤 추천 결과 및 추천 근거 ===")
    recs = recommender.generate_recommendations()
    for rec in recs:
        print(f"• 제목: {rec['title']}")
        print(f"  비율: {rec['ratios']}")
        print(f"  근거: {rec['reason']}\n")

