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
                items.append((int(w), int(h), cnt))
            except Exception:
                print(f"[ERROR] invalid format: {p}")
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

        # count が行数以下 → ユニークに使える
        if count <= total_lines:
            return indices[:count]

        # count が行数より多い → 全行 + ランダム補完
        result = indices.copy()
        extra_needed = count - total_lines
        result.extend(random.choices(indices, k=extra_needed))
        return result

    def run(self, file_path, resolution_counts, random_mode,
            negative_prompt, model, clip, vae,
            steps, cfg, sampler_name, scheduler_name,
            output_prefix):

        # ファイルチェック
        if not os.path.exists(file_path):
            print("[ERROR] File not found:", file_path)
            return (None,)

        # 行読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            original_lines = [line.strip() for line in f if line.strip()]

        total_lines = len(original_lines)
        print(f"[INFO] Loaded {total_lines} lines.")

        # 解像度解析
        res_list = self.parse_resolution_counts(resolution_counts)
        if not res_list:
            print("[ERROR] No valid resolution-count pairs")
            return (None,)
        print("[INFO] Resolutions to generate:", res_list)

        # negative prompt
        neg = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]

        last_image = None
        global_counter = 0

        # ===== resolution loop =====
        for (width, height, count) in res_list:
            print(f"[INFO] ==== Resolution {width}x{height}, Count {count} ====")

            selected_indices = self.select_lines_for_resolution(
                total_lines, count, random_mode
            )

            print(f"[INFO] Using {len(selected_indices)} lines for this resolution")

            # ===== generation loop =====
            for line_index in selected_indices:
                text = original_lines[line_index]
                print(f"[GEN] {text} (line {line_index})")

                # encode positive prompt
                cond = nodes.CLIPTextEncode().encode(clip, text)[0]

                # empty latent
                latent = nodes.EmptyLatentImage().generate(width, height, 1)[0]

                # ===== ★ 1枚ごとにランダム seed =====
                current_seed = random.randint(0, 2**32 - 1)

                # sampling
                sampler = nodes.KSampler()
                out_latent_dict = sampler.sample(
                    model,
                    current_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler_name,
                    cond,
                    neg,
                    latent
                )[0]

                latent_tensor = out_latent_dict["samples"]

                # decode
                decode_input = {"samples": latent_tensor}
                out_img = nodes.VAEDecode().decode(vae, decode_input)[0]

                # save
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
