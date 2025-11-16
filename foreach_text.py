import os
import random
import nodes

class GenerateFromTextFile:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "resolution_counts": ("STRING", {"default": "448*448:100"}),
                "random_mode": ("BOOLEAN", {"default": False}),
                "negative_prompt": ("STRING", {"default": ""}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 7.0}),
                "sampler_name": ([
                    "euler", "euler_ancestral", "heun",
                    "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "lms"
                ],),
                "scheduler_name": ([
                    "normal", "karras", "exponential", "sgm_uniform"
                ],),
                "seed": ("INT", {"default": 0}),
                "output_prefix": ("STRING", {"default": "batch_"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("last_image",)
    FUNCTION = "run"
    CATEGORY = "Batch"

    def parse_resolution_counts(self, text):
        """
        "448*448:300, 448*640:150" → [(448,448,300), (448,640,150)]
        """
        items = []
        parts = text.split(",")

        for p in parts:
            p = p.strip()
            if not p:
                continue

            try:
                res, cnt = p.split(":")
                cnt = int(cnt)

                w, h = res.split("*")
                w, h = int(w), int(h)

                items.append((w, h, cnt))

            except Exception:
                print(f"[ERROR] invalid format: {p}")
                continue

        return items

    def select_lines_for_resolution(self, total_lines, count, random_mode):
        """
        ● 解像度ごとに独立した選択を行う
        ● 同一解像度内は重複なし
        ● count > total_lines の場合は不足分をランダムに補完
        """
        indices = list(range(total_lines))

        if random_mode:
            random.shuffle(indices)
        # sequential mode →そのまま

        # 足りない場合は追加
        if count <= total_lines:
            return indices[:count]
        else:
            # まず全行をユニークに使う
            result = indices.copy()

            # 残りをランダムに再利用（重複あり）
            extra_needed = count - total_lines
            result.extend(random.choices(indices, k=extra_needed))

            return result

    def run(self, file_path, resolution_counts, random_mode,
            negative_prompt, model, clip, vae,
            steps, cfg, sampler_name, scheduler_name,
            seed, output_prefix):

        # ファイルチェック
        if not os.path.exists(file_path):
            print("[ERROR] File not found:", file_path)
            return (None,)

        # 行読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = [line.strip() for line in f if line.strip()]

        total_lines = len(original_lines)
        print(f"[INFO] Loaded {total_lines} lines.")

        # 解像度&枚数の解析
        res_list = self.parse_resolution_counts(resolution_counts)
        if not res_list:
            print("[ERROR] No valid resolution-count pairs")
            return (None,)

        print("[INFO] Resolutions to generate:", res_list)

        # ネガティブプロンプト
        neg = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]

        last_image = None
        global_counter = 0

        # ---- 各解像度ループ ----
        for (width, height, count) in res_list:
            print(f"[INFO] ==== Resolution {width}x{height}, Count {count} ====")

            # この解像度専用の行選択
            selected_indices = self.select_lines_for_resolution(
                total_lines, count, random_mode
            )

            print(f"[INFO] Using {len(selected_indices)} lines for this resolution")

            # 生成ループ
            for idx_in_res, line_index in enumerate(selected_indices):
                text = original_lines[line_index]
                print(f"[GEN] {text} (line {line_index})")

                # 正のプロンプト
                cond = nodes.CLIPTextEncode().encode(clip, text)[0]

                # 空ラティント生成
                latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]

                # KSampler
                sampler = nodes.KSampler()
                out_latent_dict = sampler.sample(
                    model,
                    seed + line_index,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler_name,
                    cond,
                    neg,
                    latent
                )[0]

                latent_tensor = out_latent_dict["samples"]

                # VAE decode
                decode_input = {"samples": latent_tensor}
                out_img = nodes.VAEDecode().decode(vae, decode_input)[0]

                global_counter += 1
                filename = f"{output_prefix}{width}x{height}_{global_counter:05d}"
                nodes.SaveImage().save_images(out_img, filename)

                last_image = out_img

        print("[DONE] All images generated.")
        return (last_image,)


NODE_CLASS_MAPPINGS = {
    "GenerateFromTextFile": GenerateFromTextFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateFromTextFile": "Generate From Text File"
}
