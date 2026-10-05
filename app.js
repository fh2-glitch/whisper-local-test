import {
    pipeline
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


/* =========================================================
   PHRASE ATTENDUE
   ========================================================= */

const TARGET_PHRASE =
    "يساعدنا الغذاء الصحي وممارسة الرياضة على النمو السليم والنجاح في دراستنا";


/* =========================================================
   PARAMÈTRES AUDIO
   ========================================================= */

const MAX_RECORDING_TIME = 10000;

const SILENCE_DURATION = 3000;

const SILENCE_THRESHOLD = 0.015;

const WHISPER_SAMPLE_RATE = 16000;


/* =========================================================
   VARIABLES
   ========================================================= */

let transcriber = null;

let busy = false;

let tempsChargementWhisper = null;

let numeroAnalyse = 0;

let useWebGPU = false;


/* =========================================================
   ÉLÉMENTS HTML
   ========================================================= */

const charger =
    document.getElementById("charger");

const parler =
    document.getElementById("parler");

const etat =
    document.getElementById("etat");

const resultat =
    document.getElementById("resultat");

const comparaison =
    document.getElementById("comparaison");

const performances =
    document.getElementById("performances");

const gpu =
    document.getElementById("gpu");


/* =========================================================
   DÉTECTION WEBGPU
   ========================================================= */

async function detectWebGPU() {

    if (!navigator.gpu) {

        gpu.textContent =
            "WebGPU disponible : NON";

        useWebGPU = false;

        return;

    }


    try {

        const adapter =
            await navigator.gpu.requestAdapter();


        if (adapter) {

            gpu.textContent =
                "WebGPU disponible : OUI — Whisper utilisera le GPU";

            useWebGPU = true;

        }

        else {

            gpu.textContent =
                "WebGPU disponible : NON";

            useWebGPU = false;

        }

    }

    catch (error) {

        console.error(
            "Erreur WebGPU :",
            error
        );


        gpu.textContent =
            "WebGPU disponible : NON";

        useWebGPU = false;

    }

}


await detectWebGPU();


/* =========================================================
   CHARGEMENT DE WHISPER
   ========================================================= */

charger.onclick = async () => {

    charger.disabled = true;


    if (!useWebGPU) {

        etat.textContent =
            "WebGPU n'est pas disponible sur cet appareil.";

        charger.disabled = false;

        return;

    }


    etat.textContent =
        "Chargement de Whisper Small avec WebGPU...";


    const debutChargement =
        performance.now();


    try {

        transcriber =
            await pipeline(

                "automatic-speech-recognition",

                "onnx-community/whisper-small",

                {
                    device: "webgpu"
                }

            );


        const finChargement =
            performance.now();


        tempsChargementWhisper =
            (
                (
                    finChargement -
                    debutChargement
                )
                / 1000
            ).toFixed(1);


        etat.textContent =
            `Whisper est prêt — WebGPU — Chargement : ${tempsChargementWhisper} s`;


        performances.innerHTML =
            `Mode : WebGPU<br>` +
            `Chargement Whisper : ${tempsChargementWhisper} s`;


        parler.disabled = false;

    }

    catch (error) {

        console.error(
            "Erreur chargement Whisper WebGPU :",
            error
        );


        etat.textContent =
            "Erreur pendant le chargement de Whisper avec WebGPU. Ouvre F12 > Console.";


        charger.disabled = false;

    }

};


/* =========================================================
   ENREGISTREMENT + WHISPER
   ========================================================= */

parler.onclick = async () => {

    if (
        !transcriber ||
        busy
    ) {

        return;

    }


    busy = true;

    parler.disabled = true;

    resultat.textContent = "—";

    comparaison.textContent = "—";


    try {

        const audio =
            await recordAudio();


        if (!audio.length) {

            etat.textContent =
                "Aucune parole détectée.";

            return;

        }


        etat.textContent =
            "Analyse de ta réponse...";


        const debutAnalyse =
            performance.now();


        const output =
            await transcriber(

                audio,

                {
                    language: "ar",
                    task: "transcribe"
                }

            );


        const finAnalyse =
            performance.now();


        const tempsAnalyse =
            (
                (
                    finAnalyse -
                    debutAnalyse
                )
                / 1000
            ).toFixed(1);


        numeroAnalyse++;


        const recognized =
            (
                output?.text ||
                ""
            ).trim();


        resultat.textContent =
            recognized ||
            "(aucun texte reconnu)";


        /* =====================================================
           NORMALISATION
           ===================================================== */

        const recognizedNormalized =
            normalizeArabic(
                recognized
            )
            .replace(
                /\s/g,
                ""
            );


        const expectedNormalized =
            normalizeArabic(
                TARGET_PHRASE
            )
            .replace(
                /\s/g,
                ""
            );


        /* =====================================================
           LEVENSHTEIN
           ===================================================== */

        const distance =
            levenshtein(

                recognizedNormalized,

                expectedNormalized

            );


        const errorRate =
            expectedNormalized.length

                ?

                distance /
                expectedNormalized.length

                :

                1;


        const errorPercent =
            Math.round(
                errorRate * 100
            );


        /* =====================================================
           RÉSULTAT
           ===================================================== */

        comparaison.textContent =

            `Distance : ${distance} — ` +

            `Erreur : ${errorPercent}% — ` +

            (
                distance <= 2

                    ?

                    "BRAVO"

                    :

                    "ESSAYE ENCORE"
            );


        /* =====================================================
           PERFORMANCES
           ===================================================== */

        etat.textContent =

            `Analyse terminée — ` +

            `WebGPU — ` +

            `Temps : ${tempsAnalyse} s`;


        performances.innerHTML =

            `Mode : WebGPU<br>` +

            `Chargement Whisper : ${tempsChargementWhisper} s<br>` +

            `Analyse n°${numeroAnalyse} : ${tempsAnalyse} s`;

    }

    catch (error) {

        console.error(
            "Erreur reconnaissance :",
            error
        );


        etat.textContent =
            "Erreur pendant l'analyse. Ouvre F12 > Console.";

    }

    finally {

        busy = false;

        parler.disabled =
            !transcriber;

    }

};


/* =========================================================
   ENREGISTREMENT AUDIO
   ========================================================= */

async function recordAudio() {

    const stream =
        await navigator.mediaDevices
            .getUserMedia({
                audio: true
            });


    const AudioContextClass =

        window.AudioContext ||

        window.webkitAudioContext;


    const audioContext =
        new AudioContextClass();


    await audioContext.resume();


    const source =
        audioContext
            .createMediaStreamSource(
                stream
            );


    const processor =
        audioContext
            .createScriptProcessor(
                4096,
                1,
                1
            );


    const chunks = [];


    let hasStartedSpeaking = false;

    let lastVoiceTime =
        Date.now();

    let finished = false;

    let maxTimer;


    etat.textContent =
        "Parle maintenant...";


    return new Promise(
        resolve => {


            function finishRecording() {

                if (finished)
                    return;


                finished = true;


                clearTimeout(
                    maxTimer
                );


                try {

                    processor.disconnect();

                    source.disconnect();

                }

                catch (_) {}


                stream
                    .getTracks()
                    .forEach(
                        track =>
                            track.stop()
                    );


                const merged =
                    mergeBuffers(
                        chunks
                    );


                const originalSampleRate =
                    audioContext.sampleRate;


                audioContext
                    .close()
                    .catch(
                        () => {}
                    );


                if (
                    !hasStartedSpeaking ||
                    !merged.length
                ) {

                    resolve(
                        new Float32Array(0)
                    );

                    return;

                }


                const audio16k =
                    resampleAudio(

                        merged,

                        originalSampleRate,

                        WHISPER_SAMPLE_RATE

                    );


                resolve(
                    audio16k
                );

            }


            processor.onaudioprocess =
                event => {


                    if (finished)
                        return;


                    const input =

                        event
                            .inputBuffer
                            .getChannelData(0);


                    chunks.push(

                        new Float32Array(
                            input
                        )

                    );


                    /* =========================================
                       VOLUME RMS
                       ========================================= */

                    let sumSquares = 0;


                    for (
                        let i = 0;
                        i < input.length;
                        i++
                    ) {

                        sumSquares +=

                            input[i] *
                            input[i];

                    }


                    const rms =
                        Math.sqrt(

                            sumSquares /

                            input.length

                        );


                    const now =
                        Date.now();


                    /* =========================================
                       PAROLE DÉTECTÉE
                       ========================================= */

                    if (
                        rms >
                        SILENCE_THRESHOLD
                    ) {

                        hasStartedSpeaking =
                            true;


                        lastVoiceTime =
                            now;

                    }


                    /* =========================================
                       SILENCE DE 3 SECONDES
                       ========================================= */

                    if (

                        hasStartedSpeaking &&

                        now -
                        lastVoiceTime >=
                        SILENCE_DURATION

                    ) {

                        finishRecording();

                    }

                };


            source.connect(
                processor
            );


            processor.connect(
                audioContext.destination
            );


            maxTimer =
                setTimeout(

                    finishRecording,

                    MAX_RECORDING_TIME

                );

        }
    );

}


/* =========================================================
   FUSION AUDIO
   ========================================================= */

function mergeBuffers(
    chunks
) {

    let totalLength = 0;


    for (
        const chunk of chunks
    ) {

        totalLength +=
            chunk.length;

    }


    const result =
        new Float32Array(
            totalLength
        );


    let offset = 0;


    for (
        const chunk of chunks
    ) {

        result.set(
            chunk,
            offset
        );


        offset +=
            chunk.length;

    }


    return result;

}


/* =========================================================
   RESAMPLING 16 kHz
   ========================================================= */

function resampleAudio(

    input,

    inputSampleRate,

    outputSampleRate

) {

    if (
        inputSampleRate ===
        outputSampleRate
    ) {

        return input;

    }


    const ratio =

        inputSampleRate /

        outputSampleRate;


    const outputLength =
        Math.round(

            input.length /

            ratio

        );


    const output =
        new Float32Array(
            outputLength
        );


    for (

        let i = 0;

        i < outputLength;

        i++

    ) {

        const position =
            i * ratio;


        const index =
            Math.floor(
                position
            );


        const fraction =
            position -
            index;


        const sample1 =
            input[index] ?? 0;


        const sample2 =

            input[
                Math.min(

                    index + 1,

                    input.length - 1

                )
            ]

            ?? sample1;


        output[i] =

            sample1 +

            (
                sample2 -
                sample1
            )

            *

            fraction;

    }


    return output;

}


/* =========================================================
   NORMALISATION ARABE
   ========================================================= */

function normalizeArabic(
    text
) {

    return text

        .replace(
            /[ًٌٍَُِّْـ]/g,
            ""
        )

        .replace(
            /[آأإٱ]/g,
            "ا"
        )

        .replace(
            /\s+/g,
            " "
        )

        .replace(
            /ة/g,
            "ه"
        )

        .replace(
            /[،؛؟.!?,:]/g,
            ""
        )

        .replace(
            /[\u200B-\u200D\uFEFF]/g,
            ""
        )

        .replace(
            /[^\u0600-\u06FF\s]/g,
            ""
        )

        .trim();

}


/* =========================================================
   LEVENSHTEIN
   ========================================================= */

function levenshtein(
    a,
    b
) {

    const matrix =
        Array.from(

            {
                length:
                    b.length + 1
            },

            () =>
                Array(
                    a.length + 1
                )

        );


    for (

        let i = 0;

        i <= b.length;

        i++

    ) {

        matrix[i][0] =
            i;

    }


    for (

        let j = 0;

        j <= a.length;

        j++

    ) {

        matrix[0][j] =
            j;

    }


    for (

        let i = 1;

        i <= b.length;

        i++

    ) {

        for (

            let j = 1;

            j <= a.length;

            j++

        ) {

            const cost =

                b[i - 1] ===
                a[j - 1]

                    ?

                    0

                    :

                    1;


            matrix[i][j] =

                Math.min(

                    matrix[i - 1][j]
                    + 1,

                    matrix[i][j - 1]
                    + 1,

                    matrix[i - 1][j - 1]
                    + cost

                );

        }

    }


    return matrix
        [b.length]
        [a.length];

}
