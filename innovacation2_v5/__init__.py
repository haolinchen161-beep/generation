"""Stage-wise structural-prior conditioning for the frozen DTGBrepGen backend."""

from .models_bbox import PriorAllocator, PriorFaceBboxModel, build_prior_face_bbox_model

__all__ = ["PriorAllocator", "PriorFaceBboxModel", "build_prior_face_bbox_model"]
