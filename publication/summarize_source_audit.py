"""Conservative evidence integration, comparisons and Arabic source-audit report."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import source_provenance_audit as audit

THRESHOLDS = dict(
    confirmed=dict(correlation=.995, ssim=.980, edge_correlation=.980, mae=.020, phash_distance=4, coverage=.85),
    probable=dict(correlation=.970, ssim=.900, edge_correlation=.900, mae=.060, phash_distance=8, coverage=.85),
    ambiguous=dict(correlation=.900, ssim=.700, phash_distance=14))
RANK = {"CONFIRMED": 3, "PROBABLE": 2, "AMBIGUOUS": 1, "NO_MATCH": 0}


def classify(frame):
    frame = frame.copy()
    def meets(threshold):
        return (frame.correlation.ge(threshold["correlation"]) & frame.ssim.ge(threshold["ssim"])
                & frame.edge_correlation.ge(threshold["edge_correlation"])
                & frame.mae.le(threshold["mae"]) & frame.phash_distance.le(threshold["phash_distance"])
                & frame.valid_coverage.ge(threshold["coverage"]) & frame.foreground_fraction.ge(.05))
    exact = frame.get("exact_pixels", pd.Series(False, index=frame.index)) | frame.get("exact_file", pd.Series(False, index=frame.index))
    confirmed, probable = exact | meets(THRESHOLDS["confirmed"]), meets(THRESHOLDS["probable"])
    ambiguous = frame.correlation.ge(.9) & frame.ssim.ge(.7) & frame.phash_distance.le(14)
    frame["category"] = np.select([confirmed, probable, ambiguous], ["CONFIRMED", "PROBABLE", "AMBIGUOUS"], default="NO_MATCH")
    frame["category_rank"] = frame.category.map(RANK)
    frame["confidence"] = frame.category.map({"CONFIRMED": "high_image_identity", "PROBABLE": "moderate_image_identity",
                                               "AMBIGUOUS": "unresolved", "NO_MATCH": "no_convincing_match_detected"})
    frame["joint"] = frame.correlation + frame.ssim + .25 * frame.edge_correlation - frame.mae
    return frame


def add_exact_evidence(frame, query, reference):
    frame = frame.copy()
    qi, ri = frame.query_index.to_numpy(int), frame.reference_index.to_numpy(int)
    frame["exact_file"] = query.iloc[qi].file_sha256.to_numpy() == reference.iloc[ri].file_sha256.to_numpy()
    frame["exact_pixels"] = False
    if "decoded_sha256" in query and "decoded_sha256" in reference:
        frame["exact_pixels"] = query.iloc[qi].decoded_sha256.to_numpy() == reference.iloc[ri].decoded_sha256.to_numpy()
    return classify(frame)


def best_rows(pairs):
    keys = ["category_rank"] + [key for key in ("exact_file", "exact_pixels") if key in pairs] + ["joint"]
    return pairs.sort_values(keys, ascending=False, kind="stable").drop_duplicates("query_index").sort_values("query_index").reset_index(drop=True)


def export_matches(pairs, query, reference, query_name, reference_name):
    rows = []
    for pair in best_rows(pairs).to_dict("records"):
        q, r = query.iloc[int(pair["query_index"])], reference.iloc[int(pair["reference_index"])]
        agreement = q.tumor_class == r.tumor_class
        row = {f"{query_name}_path": q.filepath, f"{query_name}_filename": q.filename,
               f"{query_name}_class": q.tumor_class, f"{query_name}_sample_id": q.sample_id,
               f"{reference_name}_path": r.filepath, f"{reference_name}_filename": r.filename,
               f"{reference_name}_class": r.tumor_class, f"{reference_name}_sample_id": r.sample_id,
               "class_agreement": agreement, **pair}
        for source, prefix in ((q, query_name), (r, reference_name)):
            for field in ("plane", "split", "patient_id"):
                if field in source.index: row[f"{prefix}_{field}"] = source[field]
        row["match_category"] = f"{pair['category']}_{reference_name.upper()}_MATCH" if pair["category"] in {"CONFIRMED", "PROBABLE"} else (
            "AMBIGUOUS" if pair["category"] == "AMBIGUOUS" else f"NO_{reference_name.upper()}_MATCH")
        row["notes"] = ("Shared underlying image supported; copying direction and patient independence not established."
                          if pair["category"] == "CONFIRMED" else "Nearest retrieved candidate only; identity is not confirmed.")
        if not agreement: row["notes"] += " CLASS LABEL CONFLICT: manual label review required; image match is retained."
        rows.append(row)
    return pd.DataFrame(rows)


def cross_links(indices, arrays, bc_pairs, sc_pairs, out, workers):
    destination = out / "brisc_sartaj_linked_verified_pairs.csv"
    if destination.is_file(): return pd.read_csv(destination)
    # Reuse the saved Cheng correspondences; do not rerun a 742-by-SARTAJ search.
    sc_links = sc_pairs.loc[sc_pairs.category.isin(["CONFIRMED", "PROBABLE"])]
    mapping = sc_links.groupby("reference_index").query_index.apply(lambda x: sorted(set(int(v) for v in x))).to_dict()
    focus = set(np.flatnonzero(indices["brisc_test"].screen_status.isin(["source_disjoint_candidate", "ambiguous"])))
    tasks = set()
    for row in bc_pairs.loc[bc_pairs.category.isin(["CONFIRMED", "PROBABLE"])].itertuples():
        if int(row.query_index) in focus: continue
        for s in mapping.get(int(row.reference_index), []):
            tasks.add((int(row.query_index), s))
    def work(pair):
        q, r = pair
        result = audit.verify_pair(arrays["brisc_test"][q], arrays["sartaj"][r])
        if result["ssim"] < .98:
            result = audit.sift_refine(arrays["brisc_test"][q], arrays["sartaj"][r], result)
        return dict(query_index=q, reference_index=r, retrieval_rank=0, retrieval_correlation=np.nan, **result)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, row in enumerate(pool.map(work, sorted(tasks)), 1):
            rows.append(row)
            if n % 100 == 0: print(f"Direct verification of Cheng-linked BRISC/SARTAJ pairs: {n}/{len(tasks)}", flush=True)
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["query_index", "reference_index"])
    result.to_csv(destination, index=False)
    return result


def make_visual(row, arrays, indices, qname, rname, out, stem):
    q, r = int(row.query_index), int(row.reference_index)
    qrow, rrow = indices[qname].iloc[q], indices[rname].iloc[r]
    target = arrays[qname][q, int(row.query_rep)]
    source = audit.transform(arrays[rname][r, int(row.reference_rep)], int(row.transform_code))
    warp = np.asarray(json.loads(row.warp_matrix), np.float32)
    aligned = cv2.warpAffine(source, warp, (audit.SIZE, audit.SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones(source.shape, np.uint8), warp, (audit.SIZE, audit.SIZE), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    aligned = np.clip(float(row.intensity_slope) * aligned.astype(np.float32) / 255 + float(row.intensity_offset), 0, 1)
    difference = np.abs(target.astype(np.float32) / 255 - aligned)
    difference[~valid] = np.nan
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    axes[0].imshow(audit.read_image(qrow.filepath), cmap="gray")
    axes[1].imshow(audit.read_image(rrow.filepath), cmap="gray")
    axes[2].imshow(aligned, cmap="gray", vmin=0, vmax=1)
    shown = axes[3].imshow(difference, cmap="magma", vmin=0, vmax=.20)
    titles = [f"{qname}: {qrow.filename}\n{qrow.tumor_class} | {qrow.get('plane', qrow.get('split', ''))}",
              f"{rname}: {rrow.filename}\n{rrow.tumor_class} | {rrow.get('split', '')}",
              f"Aligned {rname}\n{row.best_transform}", "Normalized difference\nInvalid warp border excluded"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.colorbar(shown, ax=axes[3], fraction=.04)
    fig.suptitle(f"{row.category} | NCC={row.correlation:.5f} | SSIM={row.ssim:.5f} | MAE={row.mae:.5f} | pHash={int(row.phash_distance)}/63\n"
                 f"{row.registration}; comparison view: query={int(row.query_rep)}, source={int(row.reference_rep)} (0=full, 1=content)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .86))
    category = {"CONFIRMED": "confirmed", "PROBABLE": "probable", "AMBIGUOUS": "ambiguous", "NO_MATCH": "no_match"}[row.category]
    folder = out / "source_audit_visuals" / category
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.png"
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(out))


def finalize(out, workers=6):
    for category in ("confirmed", "probable", "ambiguous"):
        (out / "source_audit_visuals" / category).mkdir(parents=True, exist_ok=True)
    indices, arrays = audit.load_caches(out)
    brisc, sartaj, cheng = (indices[name] for name in ("brisc_test", "sartaj", "cheng"))
    calibration = classify(pd.read_csv(out / "calibration_refined.csv"))
    assert not ((calibration.kind == "different_patient_hard_negative") & calibration.category.eq("CONFIRMED")).any(), "Threshold confirmation failed negative controls."
    calibration.to_csv(out / "calibration_classified.csv", index=False)
    audit.save_json(out / "decision_thresholds.json", dict(thresholds=THRESHOLDS,
        rationale="Precision-first conjunction inspected against observed score separation and 80 transformed positives / 80 different-patient hard negatives. No tuning to maximize matches.",
        confidence="Qualitative image-identity confidence, not a calibrated probability.",
        limitations="80 negatives do not establish a population false-match rate. Neither image matches nor absence of a detected match establishes patient independence."))
    bs_pairs = add_exact_evidence(pd.read_csv(out / "brisc_sartaj_refined_pairs.csv"), brisc, sartaj)
    bc_pairs = add_exact_evidence(pd.read_csv(out / "brisc_cheng_refined_pairs.csv"), brisc, cheng)
    sc_pairs = add_exact_evidence(pd.read_csv(out / "sartaj_cheng_refined_pairs.csv"), sartaj, cheng)
    linked = cross_links(indices, arrays, bc_pairs, sc_pairs, out, workers)
    if len(linked): bs_pairs = pd.concat([bs_pairs, add_exact_evidence(linked, brisc, sartaj)], ignore_index=True)
    for name, pairs in (("brisc_sartaj", bs_pairs), ("brisc_cheng", bc_pairs), ("sartaj_cheng", sc_pairs)):
        pairs.to_csv(out / f"{name}_classified_pairs.csv", index=False)
    bs = export_matches(bs_pairs, brisc, sartaj, "brisc", "sartaj")
    bc = export_matches(bc_pairs, brisc, cheng, "brisc", "cheng")
    sc = export_matches(sc_pairs, sartaj, cheng, "sartaj", "cheng")
    bs.to_csv(out / "brisc_sartaj_source_audit.csv", index=False)
    bc.to_csv(out / "brisc_cheng_high_precision_audit.csv", index=False)
    sc.to_csv(out / "sartaj_cheng_overlap_audit.csv", index=False)
    bs_lookup, bc_lookup, sc_lookup = (f.set_index("query_index") for f in (bs, bc, sc))
    aliases = bs_pairs.loc[bs_pairs.category == "CONFIRMED"].groupby("query_index").reference_index.apply(lambda x: sorted(set(int(i) for i in x))).to_dict()
    rows = []
    for q, b in brisc.iterrows():
        s = bs_lookup.loc[q] if q in bs_lookup.index else None
        c = bc_lookup.loc[q]
        s_status = s.category if s is not None else "NOT_SEARCHED_NO_CHENG_LINK"
        source_aliases = aliases.get(q, [])
        source_cheng_states = [sc_lookup.loc[index].category if index in sc_lookup.index else "NOT_AUDITED" for index in source_aliases]
        linked_confirmed = "CONFIRMED" in source_cheng_states
        confirmed_cheng = c.category == "CONFIRMED" or (s_status == "CONFIRMED" and linked_confirmed)
        confirmed_sartaj = s_status == "CONFIRMED"
        if confirmed_cheng and confirmed_sartaj: status = "CHENG_AND_SARTAJ_MATCH"
        elif confirmed_cheng: status = "CHENG_MATCH"
        elif confirmed_sartaj: status = "SARTAJ_MATCH"
        elif s_status == "PROBABLE": status = "PROBABLE_SARTAJ"
        elif c.category in {"PROBABLE", "AMBIGUOUS"} or s_status == "AMBIGUOUS": status = "AMBIGUOUS"
        else: status = "UNKNOWN"
        eligible = (confirmed_sartaj and c.category == "NO_MATCH" and len(source_cheng_states) > 0
                    and all(x == "NO_MATCH" for x in source_cheng_states) and bool(b.valid_mask)
                    and b.tumor_class in {"Meningioma", "Glioma", "Pituitary"} and bool(s.class_agreement))
        rows.append(dict(brisc_sample_id=b.sample_id, brisc_path=b.filepath, brisc_filename=b.filename,
            brisc_class=b.tumor_class, brisc_plane=b.plane, mask_path=b.mask_path, valid_mask=b.valid_mask,
            previous_screen_status=b.screen_status, cheng_direct_category=c.category,
            cheng_path=c.cheng_path, cheng_correlation=c.correlation, cheng_ssim=c.ssim,
            sartaj_category=s_status, sartaj_source_path=s.sartaj_path if s is not None else "",
            sartaj_split=s.sartaj_split if s is not None else "", sartaj_class=s.sartaj_class if s is not None else "",
            sartaj_match_correlation=s.correlation if s is not None else np.nan,
            sartaj_match_ssim=s.ssim if s is not None else np.nan,
            class_agreement=bool(s.class_agreement) if s is not None else False,
            sartaj_source_cheng_status="|".join(sorted(set(source_cheng_states))),
            confirmed_sartaj_aliases=json.dumps(source_aliases), final_provenance_status=status,
            clean_candidate_eligible=eligible, patient_independence_confirmed=False,
            copying_direction="not established by image matching"))
    complete = pd.DataFrame(rows)
    complete.to_csv(out / "brisc_complete_provenance_audit.csv", index=False)
    # Canonical IDs are selected lexicographically, never by masks, labels or model outcomes.
    paths = json.loads((out / "discovered_paths.json").read_text())
    hashes = pd.read_csv(Path(paths["prior_study"]) / "preprocessing_audit.csv")[["sample_id", "processed_image_hash"]]
    candidates = complete.loc[complete.clean_candidate_eligible].merge(hashes, left_on="brisc_sample_id", right_on="sample_id", validate="one_to_one")
    candidates = candidates.sort_values("brisc_sample_id")
    candidates["duplicate_group_size"] = candidates.groupby("processed_image_hash").brisc_sample_id.transform("size")
    candidates["canonical_member"] = ~candidates.processed_image_hash.duplicated()
    candidates["class"], candidates["plane"] = candidates.brisc_class, candidates.brisc_plane
    candidates["cheng_overlap_status"] = "NO_DETECTED_DIRECT_OR_SARTAJ_SOURCE_CHENG_MATCH"
    candidates.to_csv(out / "brisc_clean_candidates_before_deduplication.csv", index=False)
    candidates.loc[~candidates.canonical_member].to_csv(out / "duplicate_candidate_copies.csv", index=False)
    clean = candidates.loc[candidates.canonical_member].copy()
    clean.to_csv(out / "brisc_clean_external_candidates.csv", index=False)
    # Keep class-label conflicts even when image identity is confirmed.
    conflicts = pd.concat([bs.assign(comparison="BRISC-SARTAJ"), sc.assign(comparison="SARTAJ-Cheng")], ignore_index=True)
    conflicts.loc[conflicts.category.isin(["CONFIRMED", "PROBABLE"]) & ~conflicts.class_agreement].to_csv(out / "matched_label_conflicts.csv", index=False)
    for name, frame, groups in (("brisc_provenance_by_class_plane", complete, ["brisc_class", "brisc_plane", "final_provenance_status"]),
                                 ("sartaj_cheng_by_class_split", sc, ["sartaj_class", "sartaj_split", "category"]),
                                 ("brisc_sartaj_by_class_plane_split", bs, ["brisc_class", "brisc_plane", "sartaj_split", "category", "class_agreement"])):
        frame.groupby(groups, dropna=False).size().rename("count").reset_index().to_csv(out / f"{name}.csv", index=False)
    visual_rows = []
    focus = set(np.flatnonzero(brisc.screen_status.isin(["source_disjoint_candidate", "ambiguous"])))
    selected_bs = bs.loc[bs.query_index.isin(focus) | ((bs.category == "CONFIRMED") & ~bs.class_agreement)]
    # All 118 requested cases, plus hardest/mismatched examples for the larger audits.
    for row in selected_bs.itertuples():
        path = make_visual(row, arrays, indices, "brisc_test", "sartaj", out, f"brisc_sartaj_{int(row.query_index):04d}")
        visual_rows.append(dict(comparison="BRISC-SARTAJ", query_index=row.query_index, category=row.category, figure=path))
    for name, frame, qname, rname in (("sartaj_cheng", sc, "sartaj", "cheng"), ("brisc_cheng", bc, "brisc_test", "cheng")):
        selected = []
        for category in ("CONFIRMED", "PROBABLE", "AMBIGUOUS", "NO_MATCH"):
            subset = frame.loc[frame.category == category]
            selected.append(subset.nsmallest(4, "ssim") if category == "CONFIRMED" else subset.nlargest(4, "ssim"))
        selected.append(frame.loc[frame.category == "CONFIRMED"].groupby(f"{qname.split('_')[0]}_class", group_keys=False).head(2))
        selected.append(frame.loc[(frame.category == "CONFIRMED") & ~frame.class_agreement])
        selected = pd.concat(selected).drop_duplicates("query_index")
        for row in selected.itertuples():
            path = make_visual(row, arrays, indices, qname, rname, out, f"{name}_{int(row.query_index):04d}")
            visual_rows.append(dict(comparison=name, query_index=row.query_index, category=row.category, figure=path))
    pd.DataFrame(visual_rows).to_csv(out / "visual_review_index.csv", index=False)
    # Distributions and empirical controls justify the conservative operating region.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["correlation", "ssim", "mae"]):
        for name, data in (("SARTAJ-Cheng best", sc), ("synthetic positives", calibration.loc[calibration.kind == "synthetic_positive"]),
                            ("different-patient hard negatives", calibration.loc[calibration.kind != "synthetic_positive"])):
            ax.hist(data[metric], bins=30, histtype="step", density=True, label=name)
        ax.axvline(THRESHOLDS["confirmed"][metric], color="black", linestyle="--", label="confirmed boundary")
        ax.set(xlabel=metric, ylabel="Density")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=180)
    plt.close(fig)
    report(out, complete, bs, sc, clean, candidates, calibration)
    print("CSV outputs, comparison figures and Arabic report saved:", out, flush=True)


def report(out, complete, bs, sc, clean, candidates, calibration):
    categories = ["CHENG_MATCH", "SARTAJ_MATCH", "CHENG_AND_SARTAJ_MATCH", "PROBABLE_SARTAJ", "AMBIGUOUS", "UNKNOWN"]
    bc_counts = complete.final_provenance_status.value_counts().reindex(categories, fill_value=0)
    sc_counts = sc.category.value_counts().reindex(["CONFIRMED", "PROBABLE", "AMBIGUOUS", "NO_MATCH"], fill_value=0)
    focus = complete.loc[complete.previous_screen_status == "source_disjoint_candidate"]
    ambiguous = complete.loc[complete.previous_screen_status == "ambiguous"]
    agreements = []
    for name, frame in (("BRISC-SARTAJ", bs), ("SARTAJ-Cheng", sc)):
        for category in ("CONFIRMED", "PROBABLE"):
            group = frame.loc[frame.category == category]
            agreements.append(dict(comparison=name, category=category, same_class=int(group.class_agreement.sum()),
                                    conflicting_class=int((~group.class_agreement).sum())))
    pd.DataFrame(agreements).to_csv(out / "class_agreement_summary.csv", index=False)
    prior_focus_ids = set(complete.loc[complete.previous_screen_status.isin(["source_disjoint_candidate", "ambiguous"]), "brisc_sample_id"])
    focus_matches = bs.loc[bs.brisc_sample_id.isin(prior_focus_ids)]
    summary = dict(brisc_evaluated=len(complete), brisc_sartaj_directly_evaluated=len(bs), sartaj_tumor_evaluated=len(sc),
        brisc_provenance={k: int(v) for k, v in bc_counts.items()}, sartaj_cheng={k: int(v) for k, v in sc_counts.items()},
        original_106_confirmed_sartaj=int(focus.sartaj_category.eq("CONFIRMED").sum()),
        original_106_unknown=int(focus.final_provenance_status.eq("UNKNOWN").sum()),
        original_12_confirmed_sartaj=int(ambiguous.sartaj_category.eq("CONFIRMED").sum()),
        clean_candidates_before_dedup=len(candidates), clean_candidates_after_dedup=len(clean),
        clean_classes=clean.brisc_class.value_counts().astype(int).to_dict(), clean_planes=clean.brisc_plane.value_counts().astype(int).to_dict(),
        class_agreement_summary=agreements,
        focus_exact_files=int(focus_matches.exact_file.sum()), focus_exact_decoded_pixels=int(focus_matches.exact_pixels.sum()),
        sartaj_confirmed_with_probable_cheng=int(((complete.sartaj_category == "CONFIRMED") & complete.cheng_direct_category.eq("PROBABLE")).sum()),
        patient_independence_confirmed=False, copying_direction_established=False,
        calibration_counts=calibration.groupby(["kind", "category"]).size().rename("count").reset_index().to_dict("records"))
    audit.save_json(out / "source_audit_summary.json", summary)
    table1 = "\n".join(f"{k:30s} {int(v):5d}" for k, v in bc_counts.items())
    table2 = "\n".join(f"{k:30s} {int(v):5d}" for k, v in sc_counts.items())
    text = f"""تدقيق مصادر الصور: BRISC2025 وSARTAJ وCheng/Figshare

النطاق:
تم جرد 3264 صورة SARTAJ، منها 2764 صورة ورمية و500 No Tumor.
تم جرد 4793 زوج MRI/mask من BRISC. التدقيق الحالي يخص test الرسمي: {len(complete)} صورة.
التركيز على 106 مرشحين و12 حالة ملتبسة؛ صور train في BRISC لم تدخل المطابقة أو التقييم.
تم فحص SARTAJ الورمية كاملة مقابل 3064 صورة Cheng، مع البحث عبر جميع الكلاسات.

BRISC provenance — فئات حصرية مجموعها 860:
{table1}

SARTAJ vs Cheng — 2764 صورة ورمية:
{table2}

تنبيه لقراءة الجدول الأول: SARTAJ_MATCH يعني تأكيد الصورة في SARTAJ وعدم تأكيدها في Cheng؛ لا يعني إثبات غياب Cheng.
من هذه الحالات توجد {summary['sartaj_confirmed_with_probable_cheng']} لها Cheng overlap مرجح، ولذلك لا تدخل المرشحين النظيفين.

اتفاق تسميات الفئات:
{pd.DataFrame(agreements).to_string(index=False)}
تعارض التسمية محفوظ للمراجعة، ولا يُلغي التطابق البصري ولا يصحح الفئات تلقائيًا.

BRISC حسب الفئة:
{pd.crosstab(complete.brisc_class, complete.final_provenance_status).to_string()}

BRISC حسب MRI plane:
{pd.crosstab(complete.brisc_plane, complete.final_provenance_status).to_string()}

SARTAJ مقابل Cheng حسب الفئة وTraining/Testing:
{pd.crosstab([sc.sartaj_class, sc.sartaj_split], sc.category).to_string()}

الإجابات المباشرة:
1) من الـ106 مرشحين السابقين، {summary['original_106_confirmed_sartaj']} لديهم تطابق مؤكد مع صور في SARTAJ.
2) من الـ12 حالة الملتبسة، {summary['original_12_confirmed_sartaj']} لديهم تطابق مؤكد مع SARTAJ.
3) بقي من الـ106 عدد {summary['original_106_unknown']} مجهولي المصدر (UNKNOWN).
   ضمن الـ118: {summary['focus_exact_files']} مطابقة لبصمة الملف و{summary['focus_exact_decoded_pixels']} مطابقة للبكسلات الأصلية.
4) ثبت وجود {int(bc_counts['CHENG_AND_SARTAJ_MATCH'])} صورة BRISC لها مطابقات مؤكدة في SARTAJ وفي Cheng.
5) توجد {len(candidates)} صورة تستوفي شروط المرشحين قبل إزالة النسخ المتكررة؛ {len(clean)} بعد اختيار ممثل ثابت لكل مجموعة تكرار.
   الفئات: {json.dumps(summary['clean_classes'], ensure_ascii=False)}
   الـplanes: {json.dumps(summary['clean_planes'], ensure_ascii=False)}

القرار العلمي:
هذه مجموعة مرشحة لاختبار انتقال المصدر بعد تدقيق الصور؛ ليست مجموعة مثبتة الاستقلال على مستوى المرضى.
يمكن وصفها بأنها SARTAJ-linked, image-overlap-screened BRISC candidate subset، مع التصريح بقيودها.
لا يُثبت غياب المطابقة عدم اشتراك المريض أو السلسلة. SARTAJ ككل ليست مستقلة عن Cheng إذا وُجد تداخل أعلاه.
الفئات غير الموجودة في المجموعة لا يمكن تقييمها، ولا يصح تقديم مجموعة ذات فئة واحدة كتقييم خارجي ثلاثي الفئات.
المطابقة تثبت صورة مشتركة، ولا تثبت اتجاه النسخ تاريخيًا بين مجموعتي البيانات.

المنهج:
SHA-256 للملفات والبكسلات؛ grayscale؛ robust 1/99-percentile normalization؛ تمثيل كامل وآخر مع معالجة الحواف؛
بحث NCC بدقة 64×64 في ثمانية اتجاهات؛ أفضل ستة مرشحين؛ تحقق 256×256 بواسطة NCC وSSIM وMAE وpHash والحواف.
اختبار جميع D4 directions في البحث، ثم أفضل فرضيات المحاذاة في التحقق الدقيق.
ECC affine محدود، ثم SIFT/RANSAC محدود للحالات الأصعب. لا يوجد تشوه nonrigid ولا استخدام للـGT masks في المطابقة.
حدود التأكيد: NCC>=0.995، SSIM>=0.980، edge NCC>=0.980، MAE<=0.020، pHash<=4/63، مجال صالح>=85%.
القرار يعتمد على اجتماع المقاييس أو تطابق كامل للملف/البكسلات، لا على correlation وحده.
الحدود متحفظة ومقارنة بتوزيع النتائج وبـ80 positive controls معدلة و80 negative controls من مرضى مختلفين.
هذه controls لا تثبت معدل خطأ عام أو حساسية كاملة. النتائج السلبية تعني No detected match فقط.

العينات الـ742 السابقة لم تُعَد لها عملية بحث شاملة؛ تحققت مقاييس متعددة من Cheng IDs المحفوظة.
جرى تحقق مباشر من صور SARTAJ المرتبطة بهذه IDs عندما توفرت. غياب رابط لا يعني إثبات غياب الصورة في SARTAJ.
عدد صور BRISC التي خضعت لتحقق مباشر مع SARTAJ: {len(bs)}.
مقاييس الثقة نوعية وليست احتمالات معايرة.

النتائج التفصيلية حسب class وplane وSARTAJ split محفوظة أيضًا في ملفات CSV المنفصلة.

الصور الأصلية والأوزان وthresholds التصنيف والـsegmentation لم تتغير. لا تدريب ولا fine-tuning ولا neural network.
Torch مستخدم لضرب المصفوفات في البحث فقط؛ لم تُحمّل أوزان MAGS-Net.

المراجع المنهجية:
SSIM original authors: https://ece.uwaterloo.ca/~z70wang/research/ssim/
OpenCV ECC: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
"""
    (out / "source_audit_summary.txt").write_text(text, encoding="utf-8")
    print(table1 + "\n\n" + table2, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    finalize(args.out, args.workers)
