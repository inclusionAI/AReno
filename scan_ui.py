"""Flask web UI for the JSONL quality scanner."""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "areno" / "api"))

from flask import Flask, request, render_template_string
from quality_scanner import scan_jsonl

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AReno JSONL Quality Scanner</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f5f7fa; color: #333; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #2c3e50; margin-bottom: 8px; }
        .subtitle { color: #7f8c8d; margin-bottom: 24px; }
        .upload-box { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
                      box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .row { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
        .field { flex: 1; min-width: 200px; }
        label { display: block; font-size: 14px; color: #2c3e50; margin-bottom: 6px; font-weight: 600; }
        input[type="file"] { width: 100%; padding: 10px; border: 2px dashed #bdc3c7; border-radius: 8px; }
        input[type="text"], input[type="number"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
        .btn { background: #3498db; color: white; border: none; padding: 12px 32px; border-radius: 8px;
               font-size: 16px; cursor: pointer; width: 100%; margin-top: 8px; }
        .btn:hover { background: #2980b9; }
        .result { background: white; border-radius: 12px; padding: 24px; margin-top: 20px;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: none; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .stat-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
        .stat-card.error { background: #fff5f5; }
        .stat-card.valid { background: #f0fff4; }
        .stat-value { font-size: 28px; font-weight: 700; color: #2c3e50; }
        .stat-label { font-size: 13px; color: #7f8c8d; margin-top: 4px; }
        .bar-container { background: #ecf0f1; border-radius: 8px; height: 24px; margin: 12px 0; overflow: hidden; }
        .bar { height: 100%; border-radius: 8px; display: flex; align-items: center; justify-content: center;
               font-size: 12px; color: white; font-weight: 600; }
        .bar.valid { background: #2ecc71; }
        .bar.error { background: #e74c3c; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th { background: #2c3e50; color: white; padding: 10px; text-align: left; font-size: 13px; }
        td { padding: 10px; border-bottom: 1px solid #ecf0f1; font-size: 14px; }
        tr:hover { background: #f8f9fa; }
        .type-badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .type-blank_line { background: #ecf0f1; color: #7f8c8d; }
        .type-json_parse { background: #ffeaa7; color: #d63031; }
        .type-non_object { background: #fab1a0; color: #c0392b; }
        .type-schema_missing_field { background: #a29bfe; color: white; }
        .type-schema_empty_field { background: #fd79a8; color: white; }
        .truncated { color: #e74c3c; font-size: 13px; margin-top: 8px; }
        .pie-row { display: flex; gap: 4px; height: 30px; border-radius: 8px; overflow: hidden; margin: 16px 0; }
        .pie-seg { display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 600; }
    </style>
</head>
<body>
<div class="container">
    <h1>AReno JSONL Quality Scanner</h1>
    <p class="subtitle">上传 JSONL 文件，自动检测数据质量问题</p>

    <div class="upload-box">
        <form action="/scan" method="post" enctype="multipart/form-data">
            <div class="field">
                <label>上传 JSONL 文件</label>
                <input type="file" name="file" accept=".jsonl,.json,.txt" required>
            </div>
            <div class="row">
                <div class="field">
                    <label>必填字段（逗号分隔）</label>
                    <input type="text" name="fields" placeholder="prompt,response">
                </div>
                <div class="field" style="max-width: 150px;">
                    <label>最大错误数</label>
                    <input type="number" name="max_errors" value="100">
                </div>
            </div>
            <button type="submit" class="btn">开始扫描</button>
        </form>
    </div>

    {% if result %}
    <div class="result" style="display: block;">
        <h2>扫描结果</h2>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ result.total_lines }}</div>
                <div class="stat-label">总行数</div>
            </div>
            <div class="stat-card valid">
                <div class="stat-value" style="color: #27ae60;">{{ result.valid_records }}</div>
                <div class="stat-label">有效记录</div>
            </div>
            <div class="stat-card error">
                <div class="stat-value" style="color: #e74c3c;">{{ result.total_errors }}</div>
                <div class="stat-label">错误总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #3498db;">{{ pct }}%</div>
                <div class="stat-label">有效率</div>
            </div>
        </div>

        <div class="bar-container">
            <div class="bar valid" style="width: {{ pct }}%;">有效 {{ result.valid_records }}</div>
            <div class="bar error" style="width: {{ err_pct }}%;">错误 {{ result.total_errors }}</div>
        </div>

        <h3 style="margin-top: 20px;">错误分布</h3>
        {% if has_errors %}
        <div class="pie-row">
            {% if result.blank_lines > 0 %}<div class="pie-seg" style="background: #95a5a6; width: {{ result.blank_lines / result.total_lines * 100 }}%;">空行 {{ result.blank_lines }}</div>{% endif %}
            {% if result.json_errors > 0 %}<div class="pie-seg" style="background: #f39c12; width: {{ result.json_errors / result.total_lines * 100 }}%;">JSON {{ result.json_errors }}</div>{% endif %}
            {% if result.non_object_records > 0 %}<div class="pie-seg" style="background: #e74c3c; width: {{ result.non_object_records / result.total_lines * 100 }}%;">非对象 {{ result.non_object_records }}</div>{% endif %}
            {% if result.schema_issues > 0 %}<div class="pie-seg" style="background: #9b59b6; width: {{ result.schema_issues / result.total_lines * 100 }}%;">Schema {{ result.schema_issues }}</div>{% endif %}
            {% if result.valid_records > 0 %}<div class="pie-seg" style="background: #2ecc71; width: {{ result.valid_records / result.total_lines * 100 }}%;">有效 {{ result.valid_records }}</div>{% endif %}
        </div>
        {% endif %}

        {% if result.errors %}
        <h3 style="margin-top: 20px;">错误详情（显示前 {{ result.errors|length }} 条）</h3>
        <table>
            <tr><th>行号</th><th>错误类型</th><th>详情</th></tr>
            {% for e in result.errors %}
            <tr>
                <td>{{ e.line }}</td>
                <td><span class="type-badge type-{{ e.type }}">{{ e.type }}</span></td>
                <td>{{ e.detail if e.detail else '-' }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if result.errors_truncated > 0 %}
        <p class="truncated">还有 {{ result.errors_truncated }} 条错误未显示</p>
        {% endif %}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, result=None)


@app.route("/scan", methods=["POST"])
def scan():
    file = request.files.get("file")
    if not file:
        return render_template_string(HTML, result=None)

    # Save to temp file
    tmp_path = "/tmp/scan_upload.jsonl"
    file.save(tmp_path)

    # Parse fields
    fields_str = request.form.get("fields", "").strip()
    fields = [f.strip() for f in fields_str.split(",") if f.strip()] if fields_str else None

    max_errors = int(request.form.get("max_errors", "100") or "100")

    # Run scan
    result_obj = scan_jsonl(tmp_path, required_fields=fields, max_errors=max_errors)
    result = result_obj.to_dict()

    # Add display fields
    total = result["total_lines"] or 1
    pct = round(result["valid_records"] / total * 100, 1)
    err_pct = round(result["total_errors"] / total * 100, 1)

    os.unlink(tmp_path)

    return render_template_string(
        HTML,
        result=result,
        pct=pct,
        err_pct=err_pct,
        has_errors=result["total_errors"] > 0,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
