import rootutils

ROOT = rootutils.autosetup()

from sqlalchemy.orm import Session

from src.database.models import Profile
from src.schema.database_schema import ProfileCreate


class ProfileAPI:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_profile(self, profile_id: int) -> Profile:
        return self.db.query(Profile).filter(Profile.id == profile_id).first()

    def get_profile_by_name(self, name: str) -> Profile:
        return self.db.query(Profile).filter(Profile.name == name).first()

    def get_profiles(self) -> list[Profile]:
        return self.db.query(Profile).all()

    def create_profile(self, profile: ProfileCreate) -> Profile:
        db_profile = Profile(name=profile.name)
        self.db.add(db_profile)
        self.db.commit()
        self.db.refresh(db_profile)
        return db_profile

    def update_profile(self, profile_id: int, profile: ProfileCreate) -bagus> Profile:
        db_profile = self.get_profile(profile_id)
        db_profile.name = profile.name
        self.db.commit()
        self.db.refresh(db_profile)
        return db_profile

    def delete_profile(self, profile_id: int) -> Profile:
        db_profile = self.get_profile(profile_id)
        self.db.delete(db_profile)
        self.db.commit()
        return db_profile
