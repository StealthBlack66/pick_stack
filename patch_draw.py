import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# Replace thickness and scale in preview_until_confirm
replacements = [
    (r"cv2\.rectangle\(vis, \(x1, y1\), \(x2, y2\), \(0, 255, 0\), 2\)", r"cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)"),
    (r"cv2\.circle\(vis, d\['pixel'\], 4, \(0, 0, 255\), -1\)", r"cv2.circle(vis, d['pixel'], 2, (0, 0, 255), -1)"),
    (r"thick = 3 if d\.get\('locked'\) else 1", r"thick = 2 if d.get('locked') else 1"),
    (r"cv2\.putText\(vis,\n(\s*)f'#\{d\.get\(\"track_id\", \"\?\"\)\} \{conf:\.2f\}\{yaw_str\}\{lock_str\}',\n(\s*)\(x1, max\(15, y1 - 5\)\),\n(\s*)cv2\.FONT_HERSHEY_SIMPLEX, 0\.5, label_color, 2\)", 
     r"cv2.putText(vis,\n\g<1>f'#{d.get(\"track_id\", \"?\")} {conf:.2f}{yaw_str}{lock_str}',\n\g<2>(x1, max(15, y1 - 5)),\n\g<3>cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)"),
    (r"cv2\.putText\(vis, f'detected=\{len\(dets\)\}', \(10, 25\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.7, \(0, 255, 0\), 2\)", 
     r"cv2.putText(vis, f'detected={len(dets)}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)"),
    (r"cv2\.putText\(vis, line, \(10, y_offset\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.6, \(0, 255, 255\), 2\)", 
     r"cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)"),
    (r"cv2\.putText\(vis,\n(\s*)f'detected=\{len\(dets\)\}  ENTER=confirm  s=shot  q=quit',\n(\s*)\(10, 25\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.6, \(0, 255, 255\), 2\)", 
     r"cv2.putText(vis,\n\g<1>f'detected={len(dets)}  ENTER=confirm  s=shot  q=quit',\n\g<2>(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)")
]

for old, new in replacements:
    content = re.sub(old, new, content)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)
print("Drawing logic patched successfully!")
