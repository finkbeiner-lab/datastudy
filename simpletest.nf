#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 */

params.text = 'Hello World!'


text_ch = Channel.of(params.text)



params.outdir = "results"

log.info """\
    B A S H  H E L L O  W O R L D !
    ===================================
    bash input : ${params.text}
    """
    .stripIndent()



process BASHEX {
    tag "Bash Script Test"

    input:
    val x

    output:
    val true
    
    script:
    """
    test.sh
    """
}

workflow {
    BASHEX(text_ch)
    bashresults_ch = BASHEX.out
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
