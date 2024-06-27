"""Profile Table API"""

import rootutils

ROOT = rootutils.autosetup()

from sqlalchemy.orm import Session

from src.database.models import Profile
from src.schema.database_schema import ProfileCreate


class ProfileAPI:
    """Profile Table API"""

    def __init__(self, db: Session) -> None:
        """
        Initialize Profile Table API.

        Args:
            db (Session): Database session.

        Returns:
            None
        """
        self.db = db

    def get_profile(self, profile_id: int) -> Profile:
        """
        Get profile by ID.

        Args:
            profile_id (int): Profile ID.

        Returns:
            Profile: Profile object.
        """
        return self.db.query(Profile).filter(Profile.id == profile_id).first()

    def get_profile_by_name(self, name: str) -> Profile:
        """
        Get profile by name.

        Args:
            name (str): Profile name.

        Returns:
            Profile: Profile object.
        """
        return self.db.query(Profile).filter(Profile.name == name).first()

    def get_profiles(self) -> list[Profile]:
        """
        Get all profiles.

        Returns:
            list[Profile]: List of Profile objects.
        """
        return self.db.query(Profile).all()

    def create_profile(self, profile: ProfileCreate) -> Profile:
        """
        Create a new profile.

        Args:
            profile (ProfileCreate): Profile object.

        Returns:
            Profile: Profile object.
        """
        db_profile = Profile(name=profile.name)
        self.db.add(db_profile)
        self.db.commit()
        self.db.refresh(db_profile)
        return db_profile

    def update_profile(self, profile_id: int, profile: ProfileCreate) -> Profile:
        """
        Update profile by ID.

        Args:
            profile_id (int): Profile ID to update.
            profile (ProfileCreate): New Profile object.

        Returns:
            Profile: Profile object.
        """
        db_profile = self.get_profile(profile_id)
        db_profile.name = profile.name
        self.db.commit()
        self.db.refresh(db_profile)
        return db_profile

    def delete_profile(self, profile_id: int) -> Profile:
        """
        Delete profile by ID.

        Args:
            profile_id (int): Profile ID to delete.

        Returns:
            Profile: Profile object.
        """
        db_profile = self.get_profile(profile_id)
        self.db.delete(db_profile)
        self.db.commit()
        return db_profile
