"""Threshold decisions use validation/calibration only, never test scores."""
import numpy as np


def confusion(labels, scores, threshold=.5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions, positive = scores >= threshold, labels == 1
    tp, fp = int(np.sum(predictions & positive)), int(np.sum(predictions & ~positive))
    fn, tn = int(np.sum(~predictions & positive)), int(np.sum(~predictions & ~positive))
    recall, precision = tp / max(1, tp + fn), tp / max(1, tp + fp)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, positive_windows=tp+fn, negative_windows=tn+fp,
                recall=recall, precision=precision, false_positive_rate_per_window=fp/max(1, fp+tn),
                false_negative_rate=fn/max(1, tp+fn), f1=2*precision*recall/max(1e-12, precision+recall))


def threshold_boundaries(labels, scores, target_fpr, languages=None, groups=None):
    """Expose each negative-tail boundary, including groups with no positives."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if not 0 <= target_fpr < 1 or not np.isfinite(scores).all(): raise ValueError('Invalid calibration input')
    if labels.shape != scores.shape: raise ValueError('Calibration labels and scores must match')
    records = []
    def boundary(scope, group, selection):
        negatives = np.sort(scores[(labels == 0) & selection])[::-1]
        if not len(negatives): raise ValueError('Calibration requires negative labels')
        allowed = int(np.floor(target_fpr * len(negatives)))
        records.append(dict(scope=scope, group=group, negative_windows=len(negatives), allowed_fp=allowed,
                            threshold=float(np.nextafter(negatives[min(allowed, len(negatives)-1)], np.inf))))
    if not np.any(labels == 1): raise ValueError('Calibration requires both labels')
    boundary('overall', 'all', np.ones(len(labels), dtype=bool))
    # Prevent English/noise volume from hiding Russian false positives.
    if languages is not None:
        languages = np.asarray(languages)
        if languages.shape != labels.shape: raise ValueError('Calibration languages must match labels')
        for language in ['ru', 'en']:
            selection = languages == language
            if np.any((labels == 0) & selection): boundary('language', language, selection)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != labels.shape: raise ValueError('Calibration groups must match labels')
        for group in sorted(set(groups.tolist())):
            selection = groups == group
            if np.any((labels == 0) & selection): boundary('source', group, selection)
    return records


def calibrate(labels, scores, target_fpr, languages=None, groups=None):
    threshold = max(record['threshold'] for record in threshold_boundaries(labels, scores, target_fpr, languages, groups))
    return threshold, confusion(labels, scores, threshold)


def threshold_audit(labels, scores, target_fpr, languages, groups, constrain_sources=True):
    records = threshold_boundaries(labels, scores, target_fpr, languages, groups)
    for record in records: record['applied'] = record['scope'] != 'source' or constrain_sources
    threshold = max(record['threshold'] for record in records if record['applied'])
    for record in records: record['limits_threshold'] = record['applied'] and record['threshold'] == threshold
    return dict(threshold=threshold, source_constraints=constrain_sources, boundaries=records)


def source_group(row):
    """Keep Russian Piper and individual Silero versions visible in decisions."""
    generator = row['generator']
    rendering = row['speaker'].split('/')[0] if generator == 'silero' else generator
    return f'{row["language"]}|{generator}|{rendering}'


def operating_point(prediction, rows, target_fpr, by_language=True, by_source=False, constrain_sources=True, audit=False):
    languages = np.array([r['language'] for r in rows])
    labels, scores = np.asarray(prediction['labels']), np.asarray(prediction['scores'])
    sources = np.array([source_group(row) for row in rows]) if by_source else None
    threshold, metrics = calibrate(labels, scores, target_fpr, languages if by_language else None,
                                   sources if constrain_sources else None)
    recalls = []
    groups = {}
    for language in ['ru', 'en']:
        selected = languages == language
        if np.any((labels == 1) & selected):
            groups[language] = confusion(labels[selected], scores[selected], threshold)
            recalls.append(groups[language]['recall'])
    result = dict(threshold=threshold, metrics=metrics, groups=groups,
                  macro_recall=float(np.mean(recalls)), worst_language_recall=min(recalls))
    if by_source:
        source_metrics = {group: confusion(labels[sources == group], scores[sources == group], threshold)
                          for group in sorted(set(sources.tolist()))}
        positive_groups = [item['recall'] for item in source_metrics.values() if item['positive_windows']]
        result.update(source_groups=source_metrics, worst_source_recall=min(positive_groups))
    if audit:
        result['threshold_audit'] = threshold_audit(labels, scores, target_fpr, languages if by_language else None,
                                                   sources, constrain_sources)
    return result


def point_for_config(prediction, rows, config):
    return operating_point(prediction, rows, config['target_fpr'], by_source=config.get('source_groups', False),
                           constrain_sources=config.get('source_group_fpr_constraint', True),
                           audit=config.get('threshold_diagnostics', False))


def selection_score(point, loss, config):
    score = point['macro_recall'] + .25 * point['worst_language_recall'] - .05 * loss
    if config.get('source_groups'): score += .25 * point['worst_source_recall']
    return score
