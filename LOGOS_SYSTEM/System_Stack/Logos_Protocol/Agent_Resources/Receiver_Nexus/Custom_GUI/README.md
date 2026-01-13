# PXL Interactive Web Demo

A browser-based interactive demonstration of PXL's formal logic capabilities.

## Features

- **Real-time proposition analysis** in your browser
- **Pattern recognition** for common logical forms
- **Visual feedback** with color-coded results
- **Quick examples** for LEM, paradoxes, and modal logic
- **Responsive design** works on desktop and mobile

## Quick Start

### Option 1: Static Mode (No Backend)
Simply open `index.html` in your browser:

```bash
cd demos/interactive_web
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or just drag index.html into your browser
```

The static version provides instant pattern-based analysis without requiring a server.

### Option 2: Full Mode (With Flask Backend)
For future Coq integration:

```bash
# Install Flask (optional dependency)
pip install flask

# Start the server
python3 server.py

# Open browser to http://localhost:5000
```

## How It Works

### Frontend (`index.html` + `app.js`)
- Beautiful gradient UI with responsive design
- Textarea for proposition input
- Quick example buttons for common cases
- Real-time analysis display with syntax highlighting
- Keyboard shortcut: `Ctrl+Enter` to analyze

### Pattern Recognition
The demo recognizes:

1. **LEM Pattern**: `P \/ ~P` → Proven theorem
2. **Liar Paradox**: "This statement is false" → Ungrounded
3. **Modal Necessity**: `□(coherence(𝕆))` → Necessary truth
4. **Math Equations**: `2 + 2 = 4` → True/False with evaluation

### Backend (Optional - `server.py`)
- Flask REST API at `/api/analyze`
- Two modes: `quick` (pattern matching) and `full` (Coq verification)
- CORS enabled for development
- Status endpoint at `/api/status`

## Architecture

```
interactive_web/
├── index.html          # Main UI (standalone)
├── app.js              # Frontend logic (client-side analysis)
├── server.py           # Backend API (optional)
└── README.md           # This file
```

## Example Sessions

### Example 1: LEM
**Input**: `P \/ ~P`

**Output**:
```
Grounding: 𝕀₁ (constructively proven)
Truth Value: TRUE (theorem)

✅ This is the Law of Excluded Middle
✅ Proven constructively in PXL from trinitarian decidability
✅ Zero extra assumptions beyond 8-axiom kernel
```

### Example 2: Liar Paradox
**Input**: `This statement is false`

**Output**:
```
Grounding: UNGROUNDED
Truth Value: INDETERMINATE
Paradox Status: Resolved via privative negation

⚠️ Self-referential paradox detected
✅ Resolution: Statement lacks metaphysical grounding
   Not true, not false - simply malformed
```

### Example 3: Modal Necessity
**Input**: `□(coherence(𝕆))`

**Output**:
```
Grounding: 𝕀₃ (modal anchor)
Truth Value: NECESSARY
Modal: □ (Box) - Necessity operator

✅ Modal necessity claim detected
✅ This is the foundational axiom of PXL
```

## Technical Details

### Quick Mode (Default)
- Runs entirely in browser
- Pattern matching on common forms
- Instant results (<100ms)
- No backend required

### Full Mode (Future)
- Requires Flask backend
- Calls Coq for verification
- Actual proof construction
- Assumption analysis
- Slower but authoritative

## Browser Compatibility

Tested on:
- ✅ Chrome/Edge (Chromium) 90+
- ✅ Firefox 88+
- ✅ Safari 14+

## Keyboard Shortcuts

- `Ctrl+Enter` - Analyze proposition
- Quick example buttons - Load predefined examples

## Customization

### Add New Examples
Edit `app.js`:

```javascript
const examples = {
    custom: {
        text: "Your proposition here",
        description: "Description"
    }
};
```

Then add a button in `index.html`:
```html
<button class="example-btn" onclick="loadExample('custom')">Custom</button>
```

### Add New Patterns
Edit `analyzeLocally()` in `app.js`:

```javascript
else if (proposition.includes('your-pattern')) {
    result.grounding = "...";
    result.analysis.push("...");
}
```

## Limitations

### Current Version
- Pattern matching only (no real Coq verification in browser)
- Limited to predefined patterns
- Cannot handle arbitrary Coq syntax

### Future Enhancements
- WebAssembly Coq compilation for browser
- Full theorem proving in browser
- Interactive proof visualization
- Save/share proposition analyses
- API integration with main PXL system

## Integration

This demo can be integrated with:
- `alignment_demo.py` - For safety verification
- `lem_demo.py` - For LEM proof walkthrough
- `test_lem_discharge.py` - For kernel verification

## Development

To modify the demo:

1. Edit `index.html` for UI changes
2. Edit `app.js` for logic changes
3. Edit `server.py` for backend changes (optional)

No build step required - just refresh browser!

## Deployment

### GitHub Pages
```bash
# Copy interactive_web/ to your repo root
cp -r demos/interactive_web/* .

# Commit and push
git add index.html app.js
git commit -m "Add interactive demo"
git push

# Enable GitHub Pages in repo settings
```

### Local Server
```bash
python3 -m http.server 8000
# Open http://localhost:8000
```

### Flask Production
```bash
pip install gunicorn
gunicorn server:app -b 0.0.0.0:5000
```

## License

Part of the PXL demo suite. See main repository LICENSE.

## Support

For issues or questions:
- Main README: `/README.md`
- Demo docs: `/demos/README.md`
- Copilot instructions: `/.github/copilot-instructions.md`

---

**Status**: ✅ Complete - Fully functional browser demo  
**Version**: 1.0  
**Last Updated**: December 22, 2025
