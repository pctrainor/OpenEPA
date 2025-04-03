document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("analysis-form");
  const resultsDiv = document.getElementById("analysis-results");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // Show loading indicator
    resultsDiv.innerHTML = "<p>Analyzing data, please wait...</p>";

    const formData = {
      state: document.getElementById("state").value,
      county: document.getElementById("county").value,
      tract_id: document.getElementById("tract-id").value,
      query: document.getElementById("query").value,
    };

    // Add this after the DOMContentLoaded event handler

    // Add county selection helper
    document
      .getElementById("state")
      .addEventListener("change", async function () {
        const state = this.value;
        if (!state) return;

        try {
          const response = await fetch(
            `/counties?state=${encodeURIComponent(state)}`
          );
          const counties = await response.json();

          const countyField = document.getElementById("county");

          // Enable autocomplete for the county field
          if (counties && counties.length > 0) {
            // Clear any existing datalist
            let dataList = document.getElementById("county-list");
            if (!dataList) {
              dataList = document.createElement("datalist");
              dataList.id = "county-list";
              document.body.appendChild(dataList);
              countyField.setAttribute("list", "county-list");
            }

            // Populate datalist with counties
            dataList.innerHTML = "";
            counties.forEach((county) => {
              const option = document.createElement("option");
              option.value = county;
              dataList.appendChild(option);
            });
          }
        } catch (error) {
          console.error("Error fetching counties:", error);
        }
      });

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.error) {
        resultsDiv.innerHTML = `<p class="error">${data.error}</p>`;
      } else {
        resultsDiv.innerHTML = `
                    <h2>Analysis Results</h2>
                    <div class="analysis-text">${data.analysis.replace(
                      /\n/g,
                      "<br>"
                    )}</div>
                `;
      }
    } catch (error) {
      resultsDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  });
});
