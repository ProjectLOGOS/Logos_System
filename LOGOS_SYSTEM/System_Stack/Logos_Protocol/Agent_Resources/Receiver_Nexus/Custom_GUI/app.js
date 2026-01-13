// PXL Interactive Demo - Frontend JavaScript

const examples = {
    lem: {
        text: "P \\/ ~P",
        description: "Law of Excluded Middle"
    },
    liar: {
        text: "This statement is false",
        description: "Liar Paradox"
    },
    modal: {
        text: "□(coherence(𝕆))",
        description: "Necessary coherence of origin"
    },
    math: {
        text: "2 + 2 = 4",
        description: "Mathematical truth"
    }
};

function loadExample(exampleKey) {
    const example = examples[exampleKey];
    if (example) {
        document.getElementById('proposition').value = example.text;
        showMessage(`Loaded example: ${example.description}`, 'info');
    }
}

function clearAll() {
    document.getElementById('proposition').value = '';
    document.getElementById('result').innerHTML = `<strong>Ready for analysis</strong>

Enter a proposition above and click "Analyze" to begin.`;
    document.getElementById('result').className = 'result-box';
}

function showMessage(message, type = 'info') {
    const resultBox = document.getElementById('result');
    resultBox.className = `result-box ${type}`;
    resultBox.textContent = message;
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    loading.className = show ? 'loading active' : 'loading';
}

function analyzeProposition() {
    const proposition = document.getElementById('proposition').value.trim();
    
    if (!proposition) {
        showMessage('⚠️ Please enter a proposition first.', 'warning');
        return;
    }
    
    showLoading(true);
    
    // Simulate backend call with setTimeout
    // In production, this would be: fetch('/api/analyze', {...})
    setTimeout(() => {
        const analysis = analyzeLocally(proposition);
        displayAnalysis(analysis);
        showLoading(false);
    }, 800);
}

function analyzeLocally(proposition) {
    // Simple local analysis for demo purposes
    // Real version would call Python backend with Coq integration
    
    const result = {
        proposition: proposition,
        analysis: [],
        grounding: null,
        truthValue: null,
        modalProperties: [],
        paradoxStatus: null
    };
    
    // Detect LEM pattern
    if (proposition.match(/P\s*\\\\/\s*~\s*P/i) || 
        proposition.match(/P\s*∨\s*¬\s*P/i)) {
        result.grounding = "𝕀₁ (constructively proven)";
        result.truthValue = "TRUE (theorem)";
        result.analysis.push("✅ This is the Law of Excluded Middle");
        result.analysis.push("✅ Proven constructively in PXL from trinitarian decidability");
        result.analysis.push("✅ Zero extra assumptions beyond 8-axiom kernel");
        result.analysis.push("");
        result.analysis.push("Proof sketch:");
        result.analysis.push("  1. By trinitarian decidability:");
        result.analysis.push("     grounded_in(P, 𝕀₁) ∨ grounded_in(¬P, 𝕀₂)");
        result.analysis.push("  2. Case P in 𝕀₁: P is true, so P ∨ ¬P");
        result.analysis.push("  3. Case ¬P in 𝕀₂: ¬P is true, so P ∨ ¬P");
        result.analysis.push("  4. Therefore: P ∨ ¬P ✓");
    }
    // Detect Liar paradox
    else if (proposition.toLowerCase().includes('this statement is false') ||
             proposition.toLowerCase().includes('i am lying')) {
        result.grounding = "UNGROUNDED";
        result.truthValue = "INDETERMINATE";
        result.paradoxStatus = "Resolved via privative negation";
        result.analysis.push("⚠️ Self-referential paradox detected");
        result.analysis.push("");
        result.analysis.push("Classical analysis: Contradiction");
        result.analysis.push("  • If true → claims to be false → contradiction");
        result.analysis.push("  • If false → claim is false → must be true → contradiction");
        result.analysis.push("");
        result.analysis.push("PXL analysis: Ungrounded");
        result.analysis.push("  • Cannot ground in 𝕀₁ (truth): circular");
        result.analysis.push("  • Cannot ground in 𝕀₂ (falsity): circular");
        result.analysis.push("  • Not a modal claim for 𝕀₃");
        result.analysis.push("");
        result.analysis.push("✅ Resolution: Statement lacks metaphysical grounding");
        result.analysis.push("   Not true, not false - simply malformed");
        result.analysis.push("   Privative negation distinguishes 'not true' from 'false'");
    }
    // Detect modal necessity
    else if (proposition.includes('□') || proposition.includes('coherence(𝕆)')) {
        result.grounding = "𝕀₃ (modal anchor)";
        result.truthValue = "NECESSARY";
        result.modalProperties.push("□ (Box) - Necessity operator");
        result.analysis.push("✅ Modal necessity claim detected");
        result.analysis.push("");
        result.analysis.push("coherence(𝕆) is axiomatically given:");
        result.analysis.push("  • A7_triune_necessity: □(coherence(𝕆))");
        result.analysis.push("  • This is metaphysically necessary, not contingent");
        result.analysis.push("  • Provides foundation for all other truths");
        result.analysis.push("");
        result.analysis.push("Grounding:");
        result.analysis.push("  grounded_in(□(coherence(𝕆)), 𝕀₃)");
        result.analysis.push("  Modal properties ground in third identity anchor");
        result.analysis.push("");
        result.analysis.push("✅ This is the foundational axiom of PXL");
    }
    // Detect simple mathematical truth
    else if (proposition.match(/\d+\s*\+\s*\d+\s*=\s*\d+/)) {
        const parts = proposition.match(/(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)/);
        if (parts) {
            const a = parseInt(parts[1]);
            const b = parseInt(parts[2]);
            const claimed = parseInt(parts[3]);
            const actual = a + b;
            
            if (claimed === actual) {
                result.grounding = "𝕀₁ (truth anchor)";
                result.truthValue = "TRUE";
                result.modalProperties.push("Also: □(2+2=4) - necessarily true");
                result.analysis.push("✅ Mathematical truth verified");
                result.analysis.push("");
                result.analysis.push(`Evaluation: ${a} + ${b} = ${actual}`);
                result.analysis.push(`Claimed: ${claimed}`);
                result.analysis.push("Match: ✓");
                result.analysis.push("");
                result.analysis.push("Grounding:");
                result.analysis.push(`  grounded_in(${proposition}, 𝕀₁)`);
                result.analysis.push("  Mathematical truths ground in truth anchor");
                result.analysis.push("");
                result.analysis.push("Additionally:");
                result.analysis.push(`  grounded_in(□(${proposition}), 𝕀₃)`);
                result.analysis.push("  Mathematical truths are necessary (not contingent)");
            } else {
                result.grounding = "𝕀₂ (falsity anchor)";
                result.truthValue = "FALSE";
                result.analysis.push("❌ Mathematical statement is false");
                result.analysis.push("");
                result.analysis.push(`Evaluation: ${a} + ${b} = ${actual}`);
                result.analysis.push(`Claimed: ${claimed}`);
                result.analysis.push("Match: ✗");
                result.analysis.push("");
                result.analysis.push("Grounding:");
                result.analysis.push(`  grounded_in(¬(${proposition}), 𝕀₂)`);
                result.analysis.push("  Falsity grounds in second identity anchor");
                result.analysis.push("  Privative negation: absence of truth");
            }
        }
    }
    // Generic proposition
    else {
        result.grounding = "𝕀₁ or 𝕀₂ (to be determined)";
        result.truthValue = "UNDETERMINED (requires Coq verification)";
        result.analysis.push("ℹ️ Generic proposition entered");
        result.analysis.push("");
        result.analysis.push("To fully analyze, PXL would:");
        result.analysis.push("  1. Attempt to construct proof in Coq");
        result.analysis.push("  2. Check if proposition grounds in 𝕀₁ (true)");
        result.analysis.push("  3. Check if negation grounds in 𝕀₂ (false)");
        result.analysis.push("  4. Verify modal properties via 𝕀₃");
        result.analysis.push("");
        result.analysis.push("By trinitarian decidability:");
        result.analysis.push("  Either grounded_in(P, 𝕀₁) or grounded_in(¬P, 𝕀₂)");
        result.analysis.push("");
        result.analysis.push("💡 For full verification, use:");
        result.analysis.push("   python3 demos/alignment_demo.py");
        result.analysis.push("   (with proposition encoded as Coq theorem)");
    }
    
    return result;
}

function displayAnalysis(analysis) {
    const resultBox = document.getElementById('result');
    
    let output = `<strong>📊 Analysis Results</strong>\n\n`;
    output += `Proposition: ${analysis.proposition}\n`;
    output += `${'='.repeat(60)}\n\n`;
    
    if (analysis.grounding) {
        output += `Grounding: ${analysis.grounding}\n`;
    }
    if (analysis.truthValue) {
        output += `Truth Value: ${analysis.truthValue}\n`;
    }
    if (analysis.modalProperties.length > 0) {
        output += `Modal: ${analysis.modalProperties.join(', ')}\n`;
    }
    if (analysis.paradoxStatus) {
        output += `Paradox Status: ${analysis.paradoxStatus}\n`;
    }
    
    output += `\n${'─'.repeat(60)}\n\n`;
    output += analysis.analysis.join('\n');
    
    // Determine result box class
    let boxClass = 'result-box';
    if (analysis.truthValue === 'TRUE' || analysis.truthValue === 'TRUE (theorem)') {
        boxClass += ' success';
    } else if (analysis.truthValue === 'FALSE') {
        boxClass += ' error';
    } else if (analysis.truthValue === 'INDETERMINATE') {
        boxClass += ' warning';
    }
    
    resultBox.className = boxClass;
    resultBox.textContent = output;
}

// Keyboard shortcut: Ctrl+Enter to analyze
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('proposition').addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeProposition();
        }
    });
});
