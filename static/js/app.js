document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("transactionForm");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resetBtn = document.getElementById("resetBtn");

    const resultCard = document.getElementById("resultCard");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");

    const probabilityValue = document.getElementById("probabilityValue");
    const probabilityBar = document.getElementById("probabilityBar");

    const riskValue = document.getElementById("riskValue");
    const thresholdValue = document.getElementById("thresholdValue");
    const resultMessage = document.getElementById("resultMessage");


    /* =================================
       Form Submission
    ================================= */

    form.addEventListener("submit", async (event) => {

        event.preventDefault();

        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const dateTime =
            document.getElementById("trans_date_trans_time").value;

        if (!dateTime) {
            showError("Please select a transaction date and time.");
            return;
        }


        const transactionDate = new Date(dateTime);

        if (Number.isNaN(transactionDate.getTime())) {
            showError("Please enter a valid transaction date and time.");
            return;
        }


        /* =================================
           Loading State
        ================================= */

        setLoadingState(true);


        try {

            const unixTime = Math.floor(
                transactionDate.getTime() / 1000
            );


            /* =================================
               Transaction Data
            ================================= */

            const data = {

                trans_date_trans_time: dateTime,

                merchant:
                    document.getElementById("merchant").value.trim(),

                category:
                    document.getElementById("category").value.trim(),

                amt:
                    Number(
                        document.getElementById("amt").value
                    ),

                gender:
                    document.getElementById("gender").value,

                city:
                    document.getElementById("city").value.trim(),

                state:
                    document.getElementById("state").value.trim(),

                zip:
                    Number(
                        document.getElementById("zip").value
                    ),

                city_pop:
                    Number(
                        document.getElementById("city_pop").value
                    ),

                job:
                    document.getElementById("job").value.trim(),

                age:
                    Number(
                        document.getElementById("age").value
                    ),

                lat:
                    Number(
                        document.getElementById("lat").value
                    ),

                long:
                    Number(
                        document.getElementById("long").value
                    ),

                merch_lat:
                    Number(
                        document.getElementById("merch_lat").value
                    ),

                merch_long:
                    Number(
                        document.getElementById("merch_long").value
                    ),

                unix_time: unixTime
            };


            /* =================================
               Basic Numeric Validation
            ================================= */

            const numericFields = [
                "amt",
                "zip",
                "city_pop",
                "age",
                "lat",
                "long",
                "merch_lat",
                "merch_long"
            ];


            for (const field of numericFields) {

                if (!Number.isFinite(data[field])) {
                    throw new Error(
                        `Please enter a valid value for ${field}.`
                    );
                }

            }


            /* =================================
               Send Data to Flask
            ================================= */

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)

            });


            let result;

            try {
                result = await response.json();
            } catch {
                throw new Error(
                    "The server returned an invalid response."
                );
            }


            if (!response.ok) {

                throw new Error(
                    result.error || "Prediction failed."
                );

            }


            /* =================================
               Display Result
            ================================= */

            showResult(result);


        } catch (error) {

            showError(
                error.message || "Something went wrong."
            );


        } finally {

            setLoadingState(false);

        }

    });


    /* =================================
       Show Prediction Result
    ================================= */

    function showResult(result) {

        const probability =
            Number(result.fraud_probability);

        const threshold =
            Number(result.threshold);


        if (
            !Number.isFinite(probability) ||
            !Number.isFinite(threshold)
        ) {

            showError(
                "Invalid prediction data received from the server."
            );

            return;
        }


        const safeProbability =
            Math.min(
                Math.max(probability, 0),
                1
            );


        const percentage =
            safeProbability * 100;


        /* =================================
           Result Visibility
        ================================= */

        resultPlaceholder.style.display = "none";

        resultContent.style.display = "block";


        resultCard.className =
            "result-card result-animate";


        /* =================================
           Probability
        ================================= */

        probabilityValue.textContent =
            `${percentage.toFixed(2)}%`;


        probabilityBar.style.width =
            `${percentage}%`;


        /* =================================
           Risk
        ================================= */

        const risk =
            String(
                result.risk_level || "LOW"
            ).toUpperCase();


        riskValue.textContent =
            risk;


        riskValue.className =
            `risk-value ${risk.toLowerCase()}`;


        /* =================================
           Threshold
        ================================= */

        thresholdValue.textContent =
            `${(threshold * 100).toFixed(0)}%`;


        /* =================================
           Prediction State
        ================================= */

        if (result.prediction === "FRAUD") {

            resultCard.classList.add("fraud");

            resultMessage.textContent =
                "This transaction has been identified as potentially fraudulent and requires further review.";

        } else {

            resultCard.classList.add("legitimate");

            resultMessage.textContent =
                "This transaction appears to be legitimate based on the current detection model.";

        }

    }


    /* =================================
       Show Error
    ================================= */

    function showError(message) {

        resultPlaceholder.style.display = "none";

        resultContent.style.display = "block";


        resultCard.className =
            "result-card error";


        probabilityValue.textContent =
            "--";


        probabilityBar.style.width =
            "0%";


        riskValue.textContent =
            "ERROR";


        riskValue.className =
            "risk-value";


        thresholdValue.textContent =
            "--";


        resultMessage.textContent =
            message;

    }


    /* =================================
       Loading State
    ================================= */

    function setLoadingState(isLoading) {

        analyzeBtn.disabled =
            isLoading;


        resetBtn.disabled =
            isLoading;


        if (isLoading) {

            analyzeBtn.textContent =
                "Analyzing...";

            analyzeBtn.setAttribute(
                "aria-busy",
                "true"
            );

        } else {

            analyzeBtn.textContent =
                "Analyze Transaction";

            analyzeBtn.removeAttribute(
                "aria-busy"
            );

        }

    }


    /* =================================
       Reset
    ================================= */

    resetBtn.addEventListener("click", () => {

        form.reset();


        resultContent.style.display =
            "none";


        resultPlaceholder.style.display =
            "flex";


        resultCard.className =
            "result-card";


        probabilityValue.textContent =
            "--";


        probabilityBar.style.width =
            "0%";


        riskValue.textContent =
            "--";


        riskValue.className =
            "risk-value";


        thresholdValue.textContent =
            "90%";


        resultMessage.textContent =
            "";


        analyzeBtn.disabled =
            false;


        resetBtn.disabled =
            false;


        analyzeBtn.textContent =
            "Analyze Transaction";


        analyzeBtn.removeAttribute(
            "aria-busy"
        );

    });


    /* =================================
       Card Number Formatting
       UI ONLY — Never Sent to Backend
    ================================= */

    const cardNumber =
        document.getElementById("card_number");


    if (cardNumber) {

        cardNumber.addEventListener(
            "input",
            () => {

                let value =
                    cardNumber.value
                        .replace(/\D/g, "")
                        .slice(0, 16);


                value =
                    value.replace(
                        /(.{4})/g,
                        "$1 "
                    )
                        .trim();


                cardNumber.value =
                    value;

            }
        );

    }


    /* =================================
       Expiration Date Formatting
       UI ONLY
    ================================= */

    const expiryDate =
        document.getElementById("expiry_date");


    if (expiryDate) {

        expiryDate.addEventListener(
            "input",
            () => {

                let value =
                    expiryDate.value
                        .replace(/\D/g, "")
                        .slice(0, 4);


                if (value.length >= 3) {

                    value =
                        value.slice(0, 2)
                        + "/"
                        + value.slice(2);

                }


                expiryDate.value =
                    value;

            }
        );

    }

});