function filterTable() {
    let input = document.getElementById("searchInput");
    let filter = (input.value || "").toUpperCase().trim();
    let table = document.getElementById("patientTable");
    let tr = table.getElementsByTagName("tr");

    for (let i = 1; i < tr.length; i++) {
        const tdName = tr[i].getElementsByTagName("td")[0];
        const tdCed  = tr[i].getElementsByTagName("td")[2];

        if (tdName || tdCed) {
        const nameTxt = (tdName?.textContent || "").toUpperCase();
        const cedTxt  = (tdCed?.textContent || "").toUpperCase();
        tr[i].style.display = (nameTxt.indexOf(filter) > -1 || cedTxt.indexOf(filter) > -1) ? "" : "none";
        }
    }
}
