"""
⑨a c2pa_tagger.py — C2PA 메타데이터 서명 및 삽입

출처:
- C2PA 표준: Coalition for Content Provenance and Authenticity
  https://c2pa.org/specifications/
  → Adobe, Microsoft, BBC, Intel 등이 주도하는 콘텐츠 출처 추적 표준

- 전자서명: ES256 (ECDSA with P-256 and SHA-256)
  RFC 7518 Section 3.4
  → JSON Web Signature 표준 알고리즘

- 구현: cryptography 라이브러리
  GitHub: https://github.com/pyca/cryptography
  라이선스: Apache 2.0 / BSD

역할:
  - 쿼리 → 검색 → 합성 전 과정의 출처 이력 100% 기록
  - ES256 전자서명으로 변조 방지
  - 클립별 원본 영상 ID, 타임스탬프, 사용 위치 추적
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from ..data_models import ClipResult, C2PAManifest

logger = logging.getLogger(__name__)


class C2PATagger:
    """C2PA 메타데이터 서명 및 삽입

    build_manifest(query, sources, ts) → C2PAManifest
    sign_manifest(manifest, key) → signature (hex str)
    embed_c2pa(video_path, manifest) → .c2pa sidecar file path
    """

    def __init__(self, key_path: Optional[str] = None):
        """
        Args:
            key_path: EC Private Key (PEM) 경로
                      None이면 자동 생성 (프로토타입용)
        """
        self.key_path = key_path
        self._private_key = None
        self._public_key = None

    def _ensure_keys(self):
        """ES256 키쌍 준비"""
        if self._private_key is not None:
            return

        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization

        if self.key_path and os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
        else:
            # 프로토타입용 키 자동 생성
            self._private_key = ec.generate_private_key(ec.SECP256R1())
            logger.info("ES256 키 자동 생성 (프로토타입용)")

        self._public_key = self._private_key.public_key()

    def build_manifest(
        self,
        query: str,
        source_clips: List[ClipResult],
        timestamp: Optional[str] = None
    ) -> C2PAManifest:
        """C2PA 메타데이터 매니페스트 생성

        Args:
            query: 사용자 쿼리
            source_clips: 사용된 클립 결과 리스트
            timestamp: ISO 8601 타임스탬프 (None이면 현재 시각)

        Returns:
            C2PAManifest
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        # 클립별 출처 정보 구성
        clip_sources = []
        for i, clip in enumerate(source_clips):
            clip_info = {
                "clip_id": clip.clip_id,
                "sequence_order": i,
                "source_video": clip.video_path,
                "start_ms": clip.start_ms,
                "end_ms": clip.end_ms,
                "caption": clip.caption,
                "search_score": clip.score,
                "content_hash": self._compute_content_hash(clip)
            }
            clip_sources.append(clip_info)

        manifest = C2PAManifest(
            query=query,
            timestamp=timestamp,
            source_clips=clip_sources
        )

        logger.info(
            f"C2PA 매니페스트 생성: query='{query}', "
            f"{len(clip_sources)} clips"
        )
        return manifest

    def sign_manifest(self, manifest: C2PAManifest) -> str:
        """매니페스트에 ES256 전자서명

        Args:
            manifest: 서명할 매니페스트

        Returns:
            서명 값 (hex string)
        """
        self._ensure_keys()

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        # 매니페스트를 정규화된 JSON으로 변환
        manifest_json = json.dumps(
            manifest.to_dict(), sort_keys=True, ensure_ascii=False
        )
        manifest_bytes = manifest_json.encode("utf-8")

        # SHA-256 해시 계산
        digest = hashlib.sha256(manifest_bytes).digest()

        # ES256 서명 (ECDSA with P-256)
        signature = self._private_key.sign(
            digest,
            ec.ECDSA(hashes.SHA256())
        )

        sig_hex = signature.hex()
        manifest.signature = sig_hex

        logger.info(f"ES256 서명 완료: {sig_hex[:32]}...")
        return sig_hex

    def verify_signature(self, manifest: C2PAManifest) -> bool:
        """서명 검증

        Args:
            manifest: 검증할 매니페스트

        Returns:
            서명 유효 여부
        """
        if not manifest.signature:
            return False

        self._ensure_keys()

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        manifest_copy = C2PAManifest(
            query=manifest.query,
            timestamp=manifest.timestamp,
            source_clips=manifest.source_clips,
            signature=None
        )

        manifest_json = json.dumps(
            manifest_copy.to_dict(), sort_keys=True, ensure_ascii=False
        )
        manifest_bytes = manifest_json.encode("utf-8")
        digest = hashlib.sha256(manifest_bytes).digest()

        try:
            sig_bytes = bytes.fromhex(manifest.signature)
            self._public_key.verify(
                sig_bytes, digest, ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False

    def embed_c2pa(
        self,
        video_path: str,
        manifest: C2PAManifest,
        output_dir: Optional[str] = None
    ) -> str:
        """C2PA 매니페스트를 사이드카 파일로 저장

        프로토타입: 별도 .c2pa.json 파일로 저장
        프로덕션: 영상 파일 내부 메타데이터에 직접 삽입

        Args:
            video_path: 결과 영상 경로
            manifest: 서명된 매니페스트
            output_dir: 출력 디렉토리 (None이면 영상과 같은 디렉토리)

        Returns:
            .c2pa.json 파일 경로
        """
        if output_dir is None:
            output_dir = os.path.dirname(video_path) or "."

        os.makedirs(output_dir, exist_ok=True)

        # 영상 파일 해시 추가
        video_hash = self._file_hash(video_path) if os.path.exists(video_path) else "N/A"
        manifest_dict = manifest.to_dict()
        manifest_dict["c2pa:manifest"]["c2pa:content_hash"] = {
            "algorithm": "SHA-256",
            "hash": video_hash,
            "file": os.path.basename(video_path)
        }

        # JSON 저장
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        c2pa_path = os.path.join(output_dir, f"{base_name}.c2pa.json")

        with open(c2pa_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"C2PA 매니페스트 저장: {c2pa_path}")
        return c2pa_path

    @staticmethod
    def _compute_content_hash(clip: ClipResult) -> str:
        """클립 콘텐츠 해시 (출처 추적용)"""
        content = f"{clip.clip_id}:{clip.video_path}:{clip.start_ms}:{clip.end_ms}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _file_hash(file_path: str) -> str:
        """파일 SHA-256 해시"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
