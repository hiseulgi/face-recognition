import rootutils

ROOT = rootutils.autosetup()

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger()


def main_api(cfg: DictConfig) -> None:
    """Main entrypoint for the application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.router.recognition_router import RecognitionRouter
    from src.api.server import UvicornServer

    log.info(f"Starting API server on {cfg.api.host}:{cfg.api.port}...")

    app = FastAPI(
        title="Face Recognition API",
        description="API for face recognition",
        version=str(cfg.api.version),
        docs_url="/",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.middleware.cors.allow_origins,
        allow_credentials=cfg.api.middleware.cors.allow_credentials,
        allow_methods=cfg.api.middleware.cors.allow_methods,
        allow_headers=cfg.api.middleware.cors.allow_headers,
    )

    recognition_router = RecognitionRouter(cfg)

    app.include_router(recognition_router.router)

    server = UvicornServer(
        app=app,
        host=str(cfg.api.host),
        port=int(cfg.api.port),
        workers=int(cfg.api.workers),
    )
    server.run()


if __name__ == "__main__":
    """Main function."""

    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main entrypoint for the application."""
        if cfg.service == "api":
            main_api(cfg)
        else:
            raise ValueError(f"Unknown service: {cfg.service}")

    main()
