# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import os
import json
import csv
import math
import argparse
from pathlib import Path

PAGE_SIZE = 200

def generate_html(result_folder, output_file="result_visualization.html"):
    """
    Generate paginated HTML files to visualize results from the given folder structure.
    
    Each page shows up to 200 samples and includes navigation for previous/next pages,
    a page selector, and page count information.
    An index page is also created which contains links to all generated pages.
    
    Args:
        result_folder (str): Path to the result folder
        output_file (str): Base name for the output HTML file
    """
    # Convert to absolute path for processing
    result_folder = os.path.abspath(result_folder)
    
    # Get all sample directories (they are named with numbers)
    sample_dirs = [d for d in os.listdir(result_folder) if d.isdigit()]
    sample_dirs.sort(key=lambda x: int(x))  # Sort numerically
    
    # Load summary.csv data if available
    summary_data = {}
    summary_file = os.path.join(result_folder, "summary.csv")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'sample_id' in row:
                    sample_id = row['sample_id']
                    summary_data[sample_id] = row

    # Determine number of pages needed
    total_samples = len(sample_dirs)
    num_pages = max(1, math.ceil(total_samples / PAGE_SIZE))
    
    # Determine base name and extension for output file(s)
    base_name, ext = os.path.splitext(output_file)
    
    # Function to generate navigation HTML given the current page (1-indexed)
    def get_nav_html(current_page):
        # Determine filenames for previous and next pages (if applicable)
        if num_pages == 1:
            prev_link = next_link = ""
        else:
            prev_link = f"{base_name}_page_{current_page-1}{ext}" if current_page > 1 else ""
            next_link = f"{base_name}_page_{current_page+1}{ext}" if current_page < num_pages else ""
        
        nav = f"""
            <div class="pagination">
                <button onclick="window.location.href='{prev_link}'" {'disabled' if current_page == 1 else ''}>Previous Page</button>
                <span>Page {current_page} of {num_pages}</span>
                <button onclick="window.location.href='{next_link}'" {'disabled' if current_page == num_pages else ''}>Next Page</button>
                <input type="number" id="pageSelector" min="1" max="{num_pages}" placeholder="Page">
                <button onclick="goToPage()">Go</button>
            </div>
        """
        return nav

    # Function to generate the HTML content for a given page and its samples
    def build_html(page_samples, current_page):
        # Navigation bar (top and bottom)
        nav_html = get_nav_html(current_page)
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Result Visualization - Page {current_page} of {num_pages}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .fade-in {{
                    animation: fadeIn 0.5s ease-in;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                h1 {{
                    text-align: center;
                    color: #333;
                }}
                .pagination {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .pagination button {{
                    padding: 8px 12px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .pagination button:disabled {{
                    background-color: #ccc;
                    cursor: not-allowed;
                }}
                .pagination input[type="number"] {{
                    width: 60px;
                    padding: 6px;
                }}
                .summary-section {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .sample-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }}
                .sample {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    width: 500px;
                    margin-bottom: 20px;
                }}
                .sample-header {{
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .sample-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                }}
                .sample-content {{
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }}
                .image-container {{
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }}
                .image-title {{
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #555;
                }}
                .sample-image {{
                    width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    cursor: pointer;
                    transition: transform 0.2s;
                }}
                .sample-image:hover {{
                    transform: scale(1.02);
                }}
                .metrics-container {{
                    width: 100%;
                }}
                .metrics-summary {{
                    margin-bottom: 15px;
                }}
                .metrics-summary h3, .metrics-per-view h3 {{
                    font-size: 16px;
                    margin: 0 0 10px 0;
                    color: #333;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }}
                .metrics-cards {{
                    display: flex;
                    justify-content: space-between;
                    gap: 8px;
                }}
                .metric-card {{
                    flex: 1;
                    border-radius: 5px;
                    padding: 8px;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .summary-card {{
                    background-color: #e8f4fd;
                    color: #0277bd;
                }}
                .metric-name {{
                    font-size: 12px;
                    font-weight: bold;
                    margin-bottom: 5px;
                    color: #555;
                }}
                .metric-value {{
                    font-size: 18px;
                    font-weight: bold;
                }}
                .per-view-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }}
                .per-view-table th {{
                    background-color: #f2f2f2;
                    text-align: left;
                    padding: 6px;
                }}
                .per-view-table td {{
                    padding: 6px;
                    border-top: 1px solid #eee;
                }}
                .high-good {{
                    background-color: rgba(76, 175, 80, 0.3);
                    color: #1b5e20;
                    font-weight: bold;
                }}
                .medium-good {{
                    background-color: rgba(255, 193, 7, 0.45);
                    color: #b45f06;
                    font-weight: 500;
                }}
                .low-good {{
                    background-color: rgba(244, 67, 54, 0.2);
                    color: #c62828;
                }}
                .no-metrics, .error-metrics {{
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 3px;
                    color: #666;
                    font-style: italic;
                }}
                .error-metrics {{
                    color: #c62828;
                }}
                .video-btn {{
                    display: block;
                    background-color: #4CAF50;
                    color: white;
                    text-align: center;
                    padding: 8px 12px;
                    text-decoration: none;
                    border-radius: 4px;
                    font-weight: bold;
                    border: none;
                    cursor: pointer;
                    margin-top: 10px;
                }}
                .video-btn:hover {{
                    background-color: #45a049;
                }}
                .modal {{
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.8);
                    padding-top: 60px;
                }}
                .modal-content {{
                    margin: auto;
                    display: block;
                    max-width: 90%;
                    max-height: 90%;
                }}
                .close {{
                    position: absolute;
                    top: 15px;
                    right: 35px;
                    color: #f1f1f1;
                    font-size: 40px;
                    font-weight: bold;
                    cursor: pointer;
                }}
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                    border-radius: 5px 5px 0 0;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 10px 16px;
                    transition: 0.3s;
                    font-size: 16px;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #ccc;
                }}
                .tabcontent {{
                    display: none;
                    padding: 20px;
                    border: 1px solid #ccc;
                    border-top: none;
                    border-radius: 0 0 5px 5px;
                    animation: fadeEffect 1s;
                }}
                @keyframes fadeEffect {{
                    from {{opacity: 0;}}
                    to {{opacity: 1;}}
                }}
                .collapsible {{
                    background-color: #f8f8f8;
                    color: #444;
                    cursor: pointer;
                    padding: 12px;
                    width: 100%;
                    border: none;
                    text-align: left;
                    outline: none;
                    font-size: 15px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-radius: 4px;
                    margin-bottom: 5px;
                }}
                .collapsible:hover {{
                    background-color: #f0f0f0;
                }}
                .collapsible:after {{
                    content: '▼';
                    font-size: 13px;
                    color: #777;
                }}
                .active:after {{
                    content: '▲';
                }}
                .collapsible-content {{
                    max-height: 0;
                    overflow: hidden;
                    transition: max-height 0.2s ease-out;
                    background-color: white;
                }}
            </style>
        </head>
        <body class="fade-in">
            <div class="container">
                <h1>Result Visualization - Page {current_page} of {num_pages}</h1>
                
                <div class="tab">
                    <button class="tablinks active" onclick="openTab(event, 'AllSamples')">All Samples</button>
                    <button class="tablinks" onclick="openTab(event, 'SummaryView')">Summary</button>
                </div>
                
                <div class="legend" style="margin: 10px 0; padding: 10px; background-color: #fff; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <h3 style="margin-top: 0; font-size: 16px;">Metrics Color Legend:</h3>
                    <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center;">
                            <span style="display: inline-block; width: 20px; height: 20px; margin-right: 5px; background-color: rgba(76, 175, 80, 0.3);"></span>
                            <span>Best (top 33%)</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <span style="display: inline-block; width: 20px; height: 20px; margin-right: 5px; background-color: rgba(255, 193, 7, 0.45);"></span>
                            <span>Medium (middle 33%)</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <span style="display: inline-block; width: 20px; height: 20px; margin-right: 5px; background-color: rgba(244, 67, 54, 0.2);"></span>
                            <span>Lowest (bottom 33%)</span>
                        </div>
                        <div style="display: flex; align-items: center; margin-left: auto;">
                            <span><b>Note:</b> Metrics are compared within each column only. For PSNR and SSIM, higher is better. For LPIPS, lower is better.</span>
                        </div>
                    </div>
                </div>
                
                <div id="AllSamples" class="tabcontent" style="display: block;">
                    {nav_html}
                    <div class="sample-container">
        """
        # Add each sample for this page
        for sample_dir in page_samples:
            sample_path = os.path.join(result_folder, sample_dir)
            if not os.path.isdir(sample_path):
                continue

            # Get paths to files (relative paths for HTML)
            input_img_rel = os.path.join(sample_dir, "input.png")
            comparison_img_rel = os.path.join(sample_dir, "gt_vs_pred.png")
            video_rel = os.path.join(sample_dir, "rendered_video.mp4")
            metrics_path = os.path.join(sample_path, "metrics.json")
            
            # Read metrics if available
            metrics_html = "<div class='no-metrics'>No metrics available</div>"
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                        
                        metrics_html = "<div class='metrics-container'>"
                        # Summary section
                        if "summary" in metrics_data:
                            metrics_html += "<div class='metrics-summary'>"
                            metrics_html += "<h3>Summary Metrics</h3>"
                            metrics_html += "<div class='metrics-cards'>"
                            
                            summary = metrics_data["summary"]
                            for metric, value in summary.items():
                                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
                                metrics_html += f"<div class='metric-card summary-card'>"
                                metrics_html += f"<div class='metric-name'>{metric.upper()}</div>"
                                metrics_html += f"<div class='metric-value'>{formatted_value}</div>"
                                metrics_html += "</div>"
                            
                            metrics_html += "</div></div>"
                        
                        # Per-view section
                        if "per_view" in metrics_data and metrics_data["per_view"]:
                            metrics_html += "<div class='metrics-per-view'>"
                            metrics_html += "<button type='button' class='collapsible'>"
                            metrics_html += "<h3 style='margin: 0;'>Per-View Metrics</h3>"
                            metrics_html += "</button>"
                            metrics_html += "<div class='collapsible-content'>"
                            metrics_html += "<table class='per-view-table'>"
                            
                            # Collect all values for color coding
                            all_psnr_values = []
                            all_lpips_values = []
                            all_ssim_values = []
                            for view_data in metrics_data["per_view"]:
                                psnr = view_data.get('psnr', None)
                                lpips = view_data.get('lpips', None)
                                ssim = view_data.get('ssim', None)
                                if psnr is not None and isinstance(psnr, (int, float)):
                                    all_psnr_values.append(psnr)
                                if lpips is not None and isinstance(lpips, (int, float)):
                                    all_lpips_values.append(lpips)
                                if ssim is not None and isinstance(ssim, (int, float)):
                                    all_ssim_values.append(ssim)
                            
                            def get_color_class(value, all_values, metric_type):
                                if value is None or not isinstance(value, (int, float)):
                                    return ""
                                is_higher_better = metric_type != 'lpips'
                                if not all_values:
                                    return ""
                                sorted_values = sorted(all_values, reverse=is_higher_better)
                                if len(sorted_values) < 3:
                                    return "high-good" if value == sorted_values[0] else "low-good"
                                else:
                                    third = len(sorted_values) // 3
                                    if value in sorted_values[:third]:
                                        return "high-good"
                                    elif value in sorted_values[third:2*third]:
                                        return "medium-good"
                                    else:
                                        return "low-good"
                            
                            for view_data in metrics_data["per_view"]:
                                view_id = view_data.get("view", "N/A")
                                psnr = view_data.get('psnr', None)
                                lpips = view_data.get('lpips', None)
                                ssim = view_data.get('ssim', None)
                                psnr_display = f"{psnr:.4f}" if psnr is not None and isinstance(psnr, (int, float)) else "N/A"
                                lpips_display = f"{lpips:.4f}" if lpips is not None and isinstance(lpips, (int, float)) else "N/A"
                                ssim_display = f"{ssim:.4f}" if ssim is not None and isinstance(ssim, (int, float)) else "N/A"
                                psnr_class = get_color_class(psnr, all_psnr_values, 'psnr')
                                lpips_class = get_color_class(lpips, all_lpips_values, 'lpips')
                                ssim_class = get_color_class(ssim, all_ssim_values, 'ssim')
                                
                                metrics_html += f"<tr>"
                                metrics_html += f"<td>{view_id}</td>"
                                metrics_html += f"<td class='{psnr_class}'>{psnr_display}</td>"
                                metrics_html += f"<td class='{lpips_class}'>{lpips_display}</td>"
                                metrics_html += f"<td class='{ssim_class}'>{ssim_display}</td>"
                                metrics_html += f"</tr>"
                            
                            metrics_html += "</table></div></div>"
                        
                        metrics_html += "</div>"
                except Exception as e:
                    metrics_html = f"<div class='error-metrics'>Error reading metrics: {str(e)}</div>"
            
            # Append HTML for the sample
            html += f"""
                        <div class="sample">
                            <div class="sample-header">
                                <div class="sample-title">Sample {sample_dir}</div>
                            </div>
                            <div class="sample-content">
                                <div class="image-container">
                                    <div class="image-title">Input Image</div>
                                    <img src="{input_img_rel}" alt="Input Image" class="sample-image" 
                                         onclick="openModal('{input_img_rel}')">
                                </div>
                                <div class="image-container">
                                    <div class="image-title">GT vs Prediction</div>
                                    <img src="{comparison_img_rel}" alt="GT vs Prediction" class="sample-image" 
                                         onclick="openModal('{comparison_img_rel}')">
                                </div>
                                <div>
                                    <div class="image-title">Metrics</div>
                                    <div class="metrics">{metrics_html}</div>
                                </div>
                                <button class="video-btn" onclick="document.getElementById('videoModal').style.display='block'; 
                                         document.getElementById('videoPlayer').src='{video_rel}';">
                                    View Video
                                </button>
                            </div>
                        </div>
            """
        html += f"""
                    </div>
                    {nav_html}
                </div>
                
                <div id="SummaryView" class="tabcontent">
                    <div class="summary-section">
        """
        # Add summary view (if summary.csv exists)
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader)
                html += """
                        <h2>Summary Statistics</h2>
                        <table border="1" cellpadding="5" cellspacing="0" style="width:100%; border-collapse: collapse;">
                            <tr style="background-color: #f2f2f2;">
                """
                for header in headers:
                    html += f"<th>{header}</th>"
                html += "</tr>"
                with open(summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        html += "<tr>"
                        for header in headers:
                            html += f"<td>{row.get(header, '')}</td>"
                        html += "</tr>"
                html += "</table>"
            except Exception as e:
                html += f"<p>Error rendering summary: {str(e)}</p>"
        else:
            html += "<p>No summary.csv file found</p>"
            
        html += """
                    </div>
                </div>
                
                <!-- Image Modal -->
                <div id="imageModal" class="modal">
                    <span class="close" onclick="document.getElementById('imageModal').style.display='none'">&times;</span>
                    <img class="modal-content" id="modalImg">
                </div>
                
                <!-- Video Modal -->
                <div id="videoModal" class="modal">
                    <span class="close" onclick="document.getElementById('videoModal').style.display='none'; 
                                                 document.getElementById('videoPlayer').src='';">&times;</span>
                    <video class="modal-content" id="videoPlayer" controls autoplay></video>
                </div>
            </div>
            
            <script>
                function openModal(imgPath) {{
                    document.getElementById('imageModal').style.display = 'block';
                    document.getElementById('modalImg').src = imgPath;
                }}
                
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                // Close modal when clicking outside of the content
                window.onclick = function(event) {{
                    var imageModal = document.getElementById('imageModal');
                    var videoModal = document.getElementById('videoModal');
                    if (event.target == imageModal) {{
                        imageModal.style.display = "none";
                    }}
                    if (event.target == videoModal) {{
                        videoModal.style.display = "none";
                        document.getElementById('videoPlayer').src = '';
                    }}
                }}
                
                // Add click event for collapsible elements
                var coll = document.getElementsByClassName("collapsible");
                for (var i = 0; i < coll.length; i++) {{
                    coll[i].addEventListener("click", function() {{
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.maxHeight) {{
                            content.style.maxHeight = null;
                        }} else {{
                            content.style.maxHeight = content.scrollHeight + "px";
                        }}
                    }});
                }}
                
                // Pagination page jump function
                const totalPages = {num_pages};
                const baseName = "{base_name}";
                function goToPage() {{
                    let page = document.getElementById("pageSelector").value;
                    if(page < 1 || page > totalPages) {{
                        alert("Invalid page number");
                        return;
                    }}
                    if(totalPages === 1) {{
                        window.location.href = baseName + "{ext}";
                    }} else {{
                        window.location.href = baseName + "_page_" + page + "{ext}";
                    }}
                }}
            </script>
        </body>
        </html>
        """
        return html

    # Generate one HTML file per page
    output_files = []
    for page in range(num_pages):
        start_index = page * PAGE_SIZE
        end_index = start_index + PAGE_SIZE
        page_samples = sample_dirs[start_index:end_index]
        current_page = page + 1
        
        html_content = build_html(page_samples, current_page)
        # Determine output filename
        if num_pages == 1:
            output_path = os.path.join(result_folder, output_file)
            output_files.append(output_file)
        else:
            output_filename = f"{base_name}_page_{current_page}{ext}"
            output_path = os.path.join(result_folder, output_filename)
            output_files.append(output_filename)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML visualization created: {output_path}")
    
    # Generate an index page that lists links to all generated pages
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>Index of Generated Pages</title>
       <style>
           body {
               font-family: Arial, sans-serif;
               background-color: #f5f5f5;
               margin: 0;
               padding: 20px;
           }
           .container {
               max-width: 800px;
               margin: 0 auto;
           }
           h1 {
               text-align: center;
           }
           ul {
               list-style: none;
               padding: 0;
           }
           li {
               margin: 10px 0;
           }
           a {
               text-decoration: none;
               color: #4CAF50;
               font-size: 18px;
           }
           a:hover {
               text-decoration: underline;
           }
       </style>
    </head>
    <body>
       <div class="container">
           <h1>Index of Generated Pages</h1>
           <ul>
    """
    # Use the list of output files to generate links
    for i, file_name in enumerate(output_files, start=1):
        index_html += f'<li><a href="{file_name}">Page {i} of {num_pages}</a></li>\n'
    
    index_html += """
           </ul>
       </div>
    </body>
    </html>
    """
    
    index_path = os.path.join(result_folder, "index.html")
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"Index page created: {index_path}")
    print("All paginated HTML files and the index page have been created. You can open the index page in a web browser to view your results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paginated HTML visualization for result folder")
    parser.add_argument("result_folder", help="Path to the result folder")
    parser.add_argument("--output", "-o", default="result_visualization.html", 
                        help="Output HTML file name (default: result_visualization.html)")
    
    args = parser.parse_args()
    generate_html(args.result_folder, args.output)
