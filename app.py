import os
import tempfile
import streamlit as st

from matching import run_im_matching, run_ip_matching

st.set_page_config(page_title="iLevel KPI Alignment Platform", layout="wide")

st.title("iLevel KPI Alignment Platform")

tab_im, tab_ip = st.tabs(["Investment Memo", "Investment Proposal"])

def save_uploaded_file(uploaded_file, folder):
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ---------------- IM ----------------
with tab_im:
    st.subheader("IM Processing")

    im_files = st.file_uploader(
        "Upload IM source files",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key="im_files"
    )

    im_target = st.file_uploader(
        "Upload target workbook",
        type=["xlsx", "xlsm"],
        key="im_target"
    )

    im_threshold = st.slider(
        "Similarity threshold",
        0.0, 1.0, 0.60, 0.05,
        key="im_threshold"
    )

    if st.button("Run IM"):
        if not im_files or not im_target:
            st.warning("Please upload IM files and target file")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                im_source_paths = [
                    save_uploaded_file(f, tmpdir) for f in im_files
                ]
                im_target_path = save_uploaded_file(im_target, tmpdir)

                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)

                with st.spinner("Processing IM files..."):
                    output_file = run_im_matching(
                        input_file_source_im=im_source_paths,
                        input_file_target=im_target_path,
                        output_path=output_dir,
                        threshold=im_threshold,
                    )

                with open(output_file, "rb") as f:
                    st.download_button(
                        "Download IM Output",
                        data=f,
                        file_name=os.path.basename(output_file),
                        mime="application/vnd.ms-excel.sheet.macroEnabled.12",
                    )

# ---------------- IP ----------------
with tab_ip:
    st.subheader("IP Processing")

    ip_files = st.file_uploader(
        "Upload IP source files",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key="ip_files"
    )

    ip_target = st.file_uploader(
        "Upload target workbook",
        type=["xlsx", "xlsm"],
        key="ip_target"
    )

    ip_threshold = st.slider(
        "Similarity threshold",
        0.0, 1.0, 0.60, 0.05,
        key="ip_threshold"
    )

    if st.button("Run IP"):
        if not ip_files or not ip_target:
            st.warning("Please upload IP files and target file")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                ip_source_paths = [
                    save_uploaded_file(f, tmpdir) for f in ip_files
                ]
                ip_target_path = save_uploaded_file(ip_target, tmpdir)

                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)

                with st.spinner("Processing IP files..."):
                    output_file = run_ip_matching(
                        input_file_source_ip=ip_source_paths,
                        input_file_target=ip_target_path,
                        output_path=output_dir,
                        threshold=ip_threshold,
                    )

                with open(output_file, "rb") as f:
                    st.download_button(
                        "Download IP Output",
                        data=f,
                        file_name=os.path.basename(output_file),
                        mime="application/vnd.ms-excel.sheet.macroEnabled.12",
                    )
