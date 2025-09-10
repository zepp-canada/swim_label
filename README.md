# swim_label
# Swim Session Labeling Tool

## Controls

### Mouse
- **Hover**: move vertical cursor across plots.  
- **Right-click (x2)**: first click = start, second click = end → fills that span with the current label.  
- **Scroll**: zoom X-axis around cursor.  

### Keyboard
- **0, 1, 2, 3** → set label mode:  
  - 0 = Rest  
  - 1 = Swim  
  - 2 = Turn  
  - 3 = Dolphin  
- **← / →** → move cursor left/right by 1 sample.  
- **↑ / ↓** → load previous/next session.  
- **S** → save labels to `<session>/human_label.csv`.  
- **R** → reset labels to original (`new_swim_label.csv`).  
- **P** → toggle model prediction overlay.  
- **Esc** → discard unsaved changes and reload current session.  

### Text box (top-left)
- Enter a session index and press **Enter** to jump to that session.  

---

## Visualization Notes
- **Current labels (blue line)**: what you’re editing.  
- **Original labels (green line, shifted)**: reference labels.  
- **Predictions (orange line, shifted)**: model output.  
- **Shading**:  
  - Background colors mark your current label spans.  
  - Purple overlays highlight differences between your labels and the original.  

---

## Workflow
1. Launch the tool with your session folder root configured in `DATA_ROOT`.  
2. Browse sessions using ↑ / ↓ or jump directly by ID.  
3. Inspect the signals, yaw, and label overlays.  
4. Use right-clicks to label spans according to the correct class.  
5. Save your changes (**S**) to write `human_label.csv`.  
6. Continue to the next session.  
