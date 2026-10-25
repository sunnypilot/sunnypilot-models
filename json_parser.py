#!/usr/bin/env python3
import os
import json
import re
import urllib.parse
import argparse

def find_metadata_files(root):
  for dirpath, _, filenames in os.walk(root):
    if "metadata.json" in filenames:
      yield os.path.join(dirpath, "metadata.json")

def make_model_url(recompiled_dir, folder, file_name):
  base = f"https://huggingface.co/datasets/sunnypilot/sunnypilot_models_v1/resolve/main/models/{recompiled_dir}/"
  safe_folder = urllib.parse.quote(folder)
  safe_file = urllib.parse.quote(file_name)
  return f"{base}{safe_folder}/{safe_file}"

def update_bundle_models(bundle, meta_models, folder, recompiled_dir, allow_big_models=False):
  filtered_meta_models = [
    m for m in meta_models
    if allow_big_models or "big" not in m["artifact"]["file_name"].lower()
  ]
  updated_models = []
  for meta_model in filtered_meta_models:
    meta_copy = json.loads(json.dumps(meta_model))
    meta_copy["artifact"]["download_uri"]["url"] = make_model_url(recompiled_dir, folder, meta_model["artifact"]["file_name"])
    updated_models.append(meta_copy)
  bundle["models"] = updated_models

def collapse_overrides(json_text):
  def replacer(m):
    items = [line.strip().rstrip(',') for line in m.group(2).splitlines() if line.strip()]
    return f'{m.group(1)}{{ {", ".join(items)} }}'
  return re.sub(
    r'("overrides": ){\s*([^}]*)\s*}',
    replacer,
    json_text
  )

def get_generation_and_selector(short_name, bundles):
  candidates = [
    b for b in bundles
    if b.get("generation") not in (None, "-1", -1)
    and b.get("minimum_selector_version") not in (None, "-1", -1)
  ]
  if candidates:
    latest_bundle = max(candidates, key=lambda bundle_item: bundle_item.get("index", 0))
    return str(latest_bundle["generation"]), str(latest_bundle["minimum_selector_version"])
  return "12", "16"

def extract_date_from_display_name(display_name):
  date = re.search(r'\(([^)]+)\)', display_name)
  if not date:
    return ""
  return date.group(1)

def parse_date(date_str):
  # Try to parse "Month Day, Year" to a sortable tuple (year, month, day)
  import datetime
  try:
    return datetime.datetime.strptime(date_str, "%B %d, %Y")
  except Exception:
    return datetime.datetime.min

def main():
  parser = argparse.ArgumentParser(description="Update driving_models JSON with new recompiled models")
  parser.add_argument("--json-path", required=True, help="Path to driving_models_vX.json")
  parser.add_argument("--recompiled-dir", required=True, help="Path to recompiledX directory")
  parser.add_argument("--model-folder", required=False, help="Folder name for new model (overrides auto-detect)")
  parser.add_argument("--lat", required=False, type=str, default=".0", help="Lat smooth (decimal, e.g. 0.0)")
  parser.add_argument("--long", required=False, type=str, default=".3", help="long smooth (decimal, e.g. 0.3)")
  parser.add_argument("--generation", required=False, type=str, default=None, help="Model generation")
  parser.add_argument("--version", required=False, type=str, default=None, help="Minimum selector version")
  parser.add_argument("--set-min-version", required=False, type=str, default=None, help="Set minimum selector version for all tinygrad models")
  parser.add_argument("--sort-by-date", required=False, action="store_true", help="Sort bundles by date in display_name")
  parser.add_argument("--tinygrad-ref", required=False, type=str, default=None, help="Set tinygrad_ref top-level key in json")
  args = parser.parse_args()
  recompiled_dir_name = os.path.basename(os.path.normpath(args.recompiled_dir))

  with open(args.json_path, "r", encoding="utf-8") as f:
    driving_models_json = json.load(f)

  if args.tinygrad_ref is not None:
    driving_models_json["tinygrad_ref"] = args.tinygrad_ref

  for meta_path in find_metadata_files(args.recompiled_dir):
    with open(meta_path, "r", encoding="utf-8") as f:
      data = json.load(f)
      meta = data["bundles"][0]
    ref = meta["ref"]
    folder = os.path.basename(os.path.dirname(meta_path))
    short_name = meta.get("short_name", folder).upper()
    display_name = meta.get("display_name", short_name)

    is_big_model = meta.get("is_big")
    if is_big_model is None:
      is_big_model = any("big" in m.get("artifact", {}).get("file_name", "").lower() for m in meta.get("models", []))
    is_big_json = "chestnut" in args.json_path.lower()
    if is_big_model != is_big_json:
      continue

    bundle = None
    for b in driving_models_json["bundles"]:
      if b.get("ref") == ref:
        bundle = b
        break

    if not bundle:
      for b in driving_models_json["bundles"]:
        if b.get("short_name").upper() == short_name:
          bundle = b
          print(f"Updating bundle {short_name} with new ref: {bundle.get('ref')} -> {ref}")
          bundle["ref"] = ref
          break

    if not bundle:
      print(f"Adding new bundle for: {short_name} [ref: {ref}]")
      folder_key = args.model_folder or f"{short_name.split()[0].upper()} Models"
      index = max([bundle.get("index", 0) for bundle in driving_models_json["bundles"] if isinstance(bundle.get("index", 0), int)], default=0) + 1
      fallback_generation, fallback_version = get_generation_and_selector(short_name, driving_models_json["bundles"])
      generation = args.generation if args.generation is not None else fallback_generation
      version = args.version if args.version is not None else fallback_version

      bundle = {
        "short_name": short_name,
        "display_name": display_name,
        "is_20hz": meta.get("is_20hz", False),
        "is_big": is_big_model,
        "ref": ref,
        "environment": meta.get("environment", "development"),
        "runner": meta.get("runner", "tinygrad"),
        "index": index,
        "minimum_selector_version": version,
        "generation": generation,
        "build_time": meta.get("build_time"),
        "overrides": meta.get("overrides") or {"folder": folder_key, "lat": args.lat, "long": args.long},
        "models": []
      }
      driving_models_json["bundles"].append(bundle)

    bundle["ref"] = ref
    bundle["short_name"] = short_name
    bundle["display_name"] = display_name
    bundle["is_20hz"] = meta.get("is_20hz", bundle["is_20hz"])
    bundle["is_big"] = is_big_model
    bundle["build_time"] = meta.get("build_time", bundle.get("build_time"))
    allow_big_models = "chestnut" in args.json_path.lower()
    update_bundle_models(bundle, meta["models"], folder, recompiled_dir_name, allow_big_models=allow_big_models)

  if args.set_min_version is not None:
    for bundle in driving_models_json["bundles"]:
      bundle["minimum_selector_version"] = args.set_min_version

  if args.sort_by_date:
    def bundle_sort_key(bundle):
      date_str = extract_date_from_display_name(bundle.get("display_name", ""))
      return parse_date(date_str)
    driving_models_json["bundles"].sort(key=bundle_sort_key)
    # After sorting, arrange indexes from 0
    for idx, bundle in enumerate(driving_models_json["bundles"], 0):
      bundle["index"] = idx

  with open(args.json_path, "w", encoding="utf-8") as f:
    json_text = json.dumps(driving_models_json, indent=2)
    json_text = collapse_overrides(json_text)
    f.write(json_text)
    f.write('\n')
  print(f"{os.path.basename(args.json_path)} updated.")

if __name__ == "__main__":
  main()
