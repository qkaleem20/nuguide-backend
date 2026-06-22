"""
drive_sync.py
-------------
Pulls the Tour Guide Drive folder into the local knowledge base so ingest.py
can read it. Runs when the supervisor clicks "Sync now".
"""

import io
import json
import os
import shutil

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

# --- Config -----------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNCED_DIR = os.path.join(PROJECT_ROOT, "knowledge_base", "documents", "synced")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Drive mime types we can ingest, and the extension to save them with.
DOWNLOADABLE = {
    DOCX_MIME: ".docx",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
}

# Files whose name OTHER CODE depends on. Keyed by the Drive file's name
# (Google Docs have no extension in their Drive name). The value is the exact
# local filename it MUST be saved as. The cheat sheet is pinned in chatbot.py
# (retrieval ranker + system prompt), so its name cannot drift.
FILENAME_OVERRIDES = {
    "Academic Cheat Sheet - Extra Facts to Give on Tour": "academic_cheat_sheet.docx",
}

# Drive mime types we deliberately skip (Sheets, Slides, folders, etc.).
SKIP_MIMES = {
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.folder",
}


def _get_service():
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("DRIVE_FOLDER_ID is not set.")

    # Two ways to provide credentials:
    #  - GOOGLE_SERVICE_ACCOUNT_JSON: the key file's CONTENTS as one string.
    #    Best for Azure — set it as an app setting; nothing to upload, and the
    #    secret never lives in git.
    #  - GOOGLE_SERVICE_ACCOUNT_FILE: a path to the key file. Best locally.
    key_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    key_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

    if key_json:
        info = json.loads(key_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    elif key_file and os.path.exists(key_file):
        creds = service_account.Credentials.from_service_account_file(key_file, scopes=SCOPES)
    else:
        raise RuntimeError(
            "No Google credentials found. Set GOOGLE_SERVICE_ACCOUNT_JSON (the key "
            "file's contents) on Azure, or GOOGLE_SERVICE_ACCOUNT_FILE (a path) locally."
        )

    return build("drive", "v3", credentials=creds), folder_id


def _list_folder(service, folder_id):
    files = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, shortcutDetails(targetId, targetMimeType))",
                pageSize=200,
                pageToken=page_token,
                # supportsAllDrives flags let this work for Shared Drives too.
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _safe_name(name):
    # Strip path separators; keep it filesystem-safe but readable.
    return name.replace("/", "_").replace("\\", "_").strip()


def _target_filename(drive_name, ext):
    # Apply a forced override if this file is referenced by name elsewhere.
    if drive_name in FILENAME_OVERRIDES:
        return FILENAME_OVERRIDES[drive_name]
    base = _safe_name(drive_name)
    # Google Docs come with no extension in their name -> add the export ext.
    if not base.lower().endswith(ext):
        base = f"{base}{ext}"
    return base


def _download_to(service, file_id, out_path, export_mime=None):
    if export_mime:
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())


def sync_drive():
    """Pull the Drive folder into knowledge_base/documents/synced/.

    Returns a summary dict so the API can report what happened.
    """
    service, folder_id = _get_service()

    # Clear the synced dir so Drive deletions propagate. Local-only files live
    # in a SEPARATE folder (knowledge_base/documents/local/) and are untouched.
    if os.path.exists(SYNCED_DIR):
        shutil.rmtree(SYNCED_DIR)
    os.makedirs(SYNCED_DIR, exist_ok=True)

    files = _list_folder(service, folder_id)

    synced, skipped = [], []
    for f in files:
        name, mime, fid = f["name"], f["mimeType"], f["id"]

        # Resolve Drive shortcuts to the real file they point at. The shortcut
        # keeps its own name (so FILENAME_OVERRIDES still applies), but the
        # content and mime type come from the target.
        if mime == SHORTCUT_MIME:
            details = f.get("shortcutDetails") or {}
            fid = details.get("targetId")
            mime = details.get("targetMimeType")
            if not fid or not mime:
                skipped.append({"name": name, "reason": "shortcut has no target"})
                continue

        if mime in SKIP_MIMES:
            skipped.append({"name": name, "reason": f"unsupported type ({mime})"})
            continue

        try:
            if mime == GOOGLE_DOC_MIME:
                out_name = _target_filename(name, ".docx")
                _download_to(service, fid, os.path.join(SYNCED_DIR, out_name), export_mime=DOCX_MIME)
                synced.append(out_name)
            elif mime in DOWNLOADABLE:
                out_name = _target_filename(name, DOWNLOADABLE[mime])
                _download_to(service, fid, os.path.join(SYNCED_DIR, out_name))
                synced.append(out_name)
            else:
                skipped.append({"name": name, "reason": f"unsupported type ({mime})"})
        except Exception as e:
            # Most likely the service account can't reach the shortcut's target
            # (e.g. it lives in a Shared Drive that isn't shared with the SA).
            skipped.append({"name": name, "reason": f"could not fetch: {str(e)[:80]}"})

    return {
        "synced_count": len(synced),
        "synced_files": sorted(synced),
        "skipped": skipped,
        "synced_dir": SYNCED_DIR,
    }


if __name__ == "__main__":
    result = sync_drive()
    print(f"Synced {result['synced_count']} files to {result['synced_dir']}")
    for name in result["synced_files"]:
        print(f"  + {name}")
    for s in result["skipped"]:
        print(f"  - skipped {s['name']} ({s['reason']})")

