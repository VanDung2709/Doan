import os
import cv2
import csv
import argparse
from pathlib import Path


VIDEO_EXTENSIONS = (
    ".avi",
    ".mp4",
    ".mov",
    ".mkv"
)


def build_video_index(root_dir):
    """
    Quet toan bo thu muc DAiSEE va tao:
    ten_file -> duong_dan
    """

    index = {}

    print("\n===== DANG QUET VIDEO DAISEE =====")

    for root, dirs, files in os.walk(root_dir):

        for file in files:

            if file.lower().endswith(VIDEO_EXTENSIONS):

                full_path = os.path.join(
                    root,
                    file
                )

                index[file.lower()] = full_path


    print(
        f"Tim thay {len(index)} video."
    )

    return index


def extract_frames(
    video_path,
    output_dir,
    clip_id,
    num_frames=5
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            f"[LOI] Khong mo duoc: {video_path}"
        )

        return 0


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if total_frames <= 0:

        cap.release()
        return 0


    # Chon cac vi tri trai deu trong clip
    positions = []

    for i in range(num_frames):

        ratio = (
            (i + 1)
            / (num_frames + 1)
        )

        position = int(
            total_frames * ratio
        )

        positions.append(position)


    saved = 0


    for i, position in enumerate(positions):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            position
        )

        success, frame = cap.read()


        if not success:
            continue


        frame = cv2.resize(
            frame,
            (224, 224)
        )


        filename = (
            f"{Path(clip_id).stem}"
            f"_frame_{i+1}.jpg"
        )


        output_path = os.path.join(
            output_dir,
            filename
        )


        cv2.imwrite(
            output_path,
            frame
        )

        saved += 1


    cap.release()

    return saved


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--csv",
        required=True,
        help="File labels CSV"
    )


    parser.add_argument(
        "--videos",
        required=True,
        help="Thu muc goc chua video DAiSEE"
    )


    parser.add_argument(
        "--out",
        required=True,
        help="Thu muc output"
    )


    parser.add_argument(
        "--column",
        default="Engagement"
    )


    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=5
    )


    parser.add_argument(
        "--max_clips",
        type=int,
        default=0,
        help="0 = xu ly tat ca"
    )


    args = parser.parse_args()


    video_index = build_video_index(
        args.videos
    )


    for label in range(4):

        os.makedirs(
            os.path.join(
                args.out,
                str(label)
            ),
            exist_ok=True
        )


    total_clips = 0
    total_frames = 0
    missing = 0


    with open(
        args.csv,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)


        print(
            "\n===== COT TRONG CSV ====="
        )

        print(
            reader.fieldnames
        )


        for row in reader:

            clip_id = (
                row.get("ClipID")
                or row.get("ClipId")
                or row.get("clip_id")
                or row.get("Clip")
            )


            if not clip_id:

                print(
                    "[LOI] Khong tim thay cot ClipID."
                )

                break


            label = int(
                row[args.column]
            )


            video_path = video_index.get(
                clip_id.lower()
            )


            # Thu them extension neu CSV khong co
            if video_path is None:

                stem = Path(
                    clip_id
                ).stem.lower()


                for ext in VIDEO_EXTENSIONS:

                    key = stem + ext

                    if key in video_index:

                        video_path = (
                            video_index[key]
                        )

                        break


            if video_path is None:

                missing += 1

                print(
                    f"[THIEU] {clip_id}"
                )

                continue


            class_dir = os.path.join(
                args.out,
                str(label)
            )


            saved = extract_frames(
                video_path,
                class_dir,
                clip_id,
                args.frames_per_clip
            )


            total_frames += saved
            total_clips += 1


            print(
                f"[OK] {clip_id}"
                f" | Engagement={label}"
                f" | frames={saved}"
            )


            if (
                args.max_clips > 0
                and
                total_clips >= args.max_clips
            ):

                break


    print(
        "\n=============================="
    )

    print(
        "KET QUA TRICH FRAME DAISEE"
    )

    print(
        "=============================="
    )

    print(
        f"Clip xu ly thanh cong: "
        f"{total_clips}"
    )

    print(
        f"Tong frame da luu: "
        f"{total_frames}"
    )

    print(
        f"Clip khong tim thay: "
        f"{missing}"
    )

    print(
        f"Output: {args.out}"
    )


if __name__ == "__main__":
    main()