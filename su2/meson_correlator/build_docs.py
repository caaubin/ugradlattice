#!/usr/bin/env python3
"""
Documentation build system for Lattice QCD Modular System

This script automates the complete documentation generation process:
1. Builds HTML documentation with Sphinx
2. Attempts PDF generation if LaTeX is available
3. Creates a comprehensive index of all outputs
4. Organizes files for GitHub Pages deployment

Usage:
    python3 build_docs.py [--clean] [--html] [--pdf] [--all]
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, description, ignore_errors=False):
    """Run a command and handle errors gracefully"""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️ {description} failed (ignoring): {e.stderr[:100]}")
            return False
        else:
            print(f"❌ {description} failed: {e.stderr[:100]}")
            return False

def clean_build_directories():
    """Clean previous build outputs"""
    print("🧹 Cleaning previous builds...")

    dirs_to_clean = [
        "documentation/_build",
        "documentation/_autosummary",
        "docs"  # GitHub Pages output
    ]

    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")

    print("✅ Cleanup completed")

def build_html_docs():
    """Build HTML documentation using Sphinx"""
    print("📚 Building HTML documentation...")

    os.chdir("documentation")
    success = run_command("make html", "HTML documentation build")
    os.chdir("..")

    if success:
        print("✅ HTML documentation built successfully")
        print("   📁 Output: documentation/_build/html/")
        return True
    return False

def build_pdf_docs():
    """Attempt to build PDF documentation"""
    print("📄 Attempting PDF documentation build...")

    # Check if LaTeX is available
    latex_available = run_command("which pdflatex", "LaTeX availability check", ignore_errors=True)

    if not latex_available:
        print("⚠️ LaTeX not available, skipping PDF generation")
        print("   Install texlive-latex-base and texlive-latex-recommended for PDF support")
        return False

    os.chdir("documentation")
    success = run_command("make latexpdf", "PDF documentation build", ignore_errors=True)
    os.chdir("..")

    if success:
        print("✅ PDF documentation built successfully")
        print("   📁 Output: documentation/_build/latex/LatticeQCDModularSystem.pdf")
        return True
    else:
        print("⚠️ PDF generation failed (LaTeX issues are common)")
        return False

def create_github_pages_structure():
    """Create GitHub Pages compatible structure"""
    print("🌐 Creating GitHub Pages structure...")

    # Create docs directory for GitHub Pages
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # Copy HTML documentation
    html_source = Path("documentation/_build/html")
    if html_source.exists():
        # Copy all HTML files
        for item in html_source.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(html_source)
                dest_path = docs_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)

        print("✅ HTML files copied to docs/")

    # Copy PDF if available
    pdf_source = Path("documentation/_build/latex/LatticeQCDModularSystem.pdf")
    if pdf_source.exists():
        shutil.copy2(pdf_source, docs_dir / "LatticeQCDModularSystem.pdf")
        print("✅ PDF copied to docs/")

    # Create .nojekyll for GitHub Pages
    (docs_dir / ".nojekyll").touch()

    # Create simple redirect index
    create_docs_index(docs_dir)

    print("✅ GitHub Pages structure created")
    print("   📁 Output: docs/ (ready for GitHub Pages)")

def create_docs_index(docs_dir):
    """Create an index file for the docs directory"""
    index_content = """<!DOCTYPE html>
<html>
<head>
    <title>Lattice QCD Modular System Documentation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: #2980B9; }
        .docs-link {
            display: inline-block;
            padding: 10px 20px;
            background: #3498DB;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }
        .docs-link:hover { background: #2980B9; }
        .description { color: #666; margin-bottom: 30px; }
    </style>
</head>
<body>
    <h1>🔬 Lattice QCD Modular System Documentation</h1>

    <p class="description">
        A modular, high-performance lattice QCD propagator system for meson mass calculations
        using Wilson fermions on discrete spacetime lattices.
    </p>

    <h2>📚 Documentation Access</h2>

    <p>
        <a href="index.html" class="docs-link">📖 Browse Full Documentation</a><br>
        <small>Complete HTML documentation with API reference, tutorials, and examples</small>
    </p>

    <p>
        <a href="LatticeQCDModularSystem.pdf" class="docs-link">📄 Download PDF Manual</a><br>
        <small>Offline-ready PDF version of the complete documentation</small>
    </p>

    <h2>🚀 Quick Start</h2>

    <pre><code># Calculate pion mass on 4×4×4×4 lattice
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel pion \\
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_pion

# Calculate full meson spectrum
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel all \\
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_spectrum</code></pre>

    <h2>📋 System Components</h2>
    <ul>
        <li><strong>PropagatorModular.py</strong> - Main CLI interface (backward compatible)</li>
        <li><strong>MesonIntegration.py</strong> - Unified coordination layer</li>
        <li><strong>MesonBase.py</strong> - Shared infrastructure and algorithms</li>
        <li><strong>PionCalculator.py</strong> - Pseudoscalar meson calculations</li>
        <li><strong>RhoCalculator.py</strong> - Vector meson calculations</li>
        <li><strong>SigmaCalculator.py</strong> - Scalar meson calculations</li>
        <li><strong>test_modules.py</strong> - Comprehensive validation framework</li>
    </ul>

    <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
        <p><small>
            Generated by the Lattice QCD Modular System Documentation Builder<br>
            Author: Zeke Mohammed | Advisor: Dr. Aubin | Institution: Fordham University
        </small></p>
    </footer>
</body>
</html>"""

    with open(docs_dir / "index.html", "w") as f:
        f.write(index_content)

def print_build_summary():
    """Print a summary of all generated outputs"""
    print("\n" + "="*60)
    print("📊 BUILD SUMMARY")
    print("="*60)

    outputs = []

    # Check HTML output
    if os.path.exists("documentation/_build/html/index.html"):
        outputs.append("✅ HTML Documentation: documentation/_build/html/")

    # Check PDF output
    if os.path.exists("documentation/_build/latex/LatticeQCDModularSystem.pdf"):
        outputs.append("✅ PDF Documentation: documentation/_build/latex/LatticeQCDModularSystem.pdf")

    # Check GitHub Pages output
    if os.path.exists("docs/index.html"):
        outputs.append("✅ GitHub Pages: docs/ (ready for deployment)")

    # Check sample inputs
    if os.path.exists("sample_inputs/metadata.json"):
        outputs.append("✅ Sample Configurations: sample_inputs/")

    # Check expected outputs doc
    if os.path.exists("documentation/expected_outputs.md"):
        outputs.append("✅ Expected Outputs Guide: documentation/expected_outputs.md")

    for output in outputs:
        print(f"   {output}")

    print(f"\n🎯 Total outputs generated: {len(outputs)}")

    if os.path.exists("docs/index.html"):
        print("\n🌐 GitHub Pages Setup:")
        print("   1. Push to GitHub")
        print("   2. Go to Settings > Pages")
        print("   3. Select 'Deploy from a branch'")
        print("   4. Choose 'main' branch and '/docs' folder")
        print("   5. Your docs will be available at: https://username.github.io/repo-name/")

def main():
    """Main build orchestration"""
    parser = argparse.ArgumentParser(description="Build Lattice QCD documentation")
    parser.add_argument("--clean", action="store_true", help="Clean previous builds")
    parser.add_argument("--html", action="store_true", help="Build HTML only")
    parser.add_argument("--pdf", action="store_true", help="Build PDF only")
    parser.add_argument("--all", action="store_true", help="Build everything (default)")

    args = parser.parse_args()

    # Default to --all if no specific options
    if not any([args.html, args.pdf]):
        args.all = True

    print("🔧 LATTICE QCD DOCUMENTATION BUILDER")
    print("="*50)

    # Clean if requested
    if args.clean or args.all:
        clean_build_directories()

    success_count = 0

    # Build HTML
    if args.html or args.all:
        if build_html_docs():
            success_count += 1

    # Build PDF
    if args.pdf or args.all:
        if build_pdf_docs():
            success_count += 1

    # Create GitHub Pages structure
    if args.all:
        create_github_pages_structure()
        success_count += 1

    # Print summary
    print_build_summary()

    if success_count > 0:
        print(f"\n🎉 Documentation build completed successfully!")
        print(f"   View HTML docs: open documentation/_build/html/index.html")
        if os.path.exists("docs/index.html"):
            print(f"   GitHub Pages ready: docs/ directory")
    else:
        print(f"\n❌ Documentation build failed")
        sys.exit(1)

if __name__ == "__main__":
    main()