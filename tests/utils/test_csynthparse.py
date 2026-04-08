from pathlib import Path

import pandas as pd

from pysilicon.utils.csynthparse import CsynthParser


def test_get_loop_pipeline_info_extracts_trip_counts_and_latency_range(tmp_path: Path):
    report_dir = tmp_path / "syn" / "report"
    report_dir.mkdir(parents=True)
    report_xml = report_dir / "csynth.xml"
    report_xml.write_text(
        """
<Report>
  <ModuleInformation>
    <Module>
      <Name>poly</Name>
      <PerformanceEstimates>
        <SummaryOfLoopLatency>
          <compute_loop>
            <Name>compute_loop</Name>
            <TripCount>
              <range>
                <min>1</min>
                <max>512</max>
              </range>
            </TripCount>
            <Latency>30 ~ 541</Latency>
            <PipelineII>1</PipelineII>
            <PipelineDepth>30</PipelineDepth>
          </compute_loop>
          <drain_loop>
            <Name>drain_loop</Name>
            <Latency>7</Latency>
            <PipelineII>2</PipelineII>
            <PipelineDepth>4</PipelineDepth>
          </drain_loop>
        </SummaryOfLoopLatency>
      </PerformanceEstimates>
    </Module>
  </ModuleInformation>
</Report>
""".strip(),
        encoding="utf-8",
    )

    parser = CsynthParser(sol_path=str(tmp_path))
    parser.get_loop_pipeline_info()

    assert list(parser.loop_df.columns) == [
      "PipelineII",
      "PipelineDepth",
        "TripCountMin",
        "TripCountMax",
        "LatencyMin",
        "LatencyMax",
    ]

    compute_loop = parser.loop_df.loc["poly:compute_loop"]
    assert compute_loop["TripCountMin"] == 1
    assert compute_loop["TripCountMax"] == 512
    assert compute_loop["LatencyMin"] == 30
    assert compute_loop["LatencyMax"] == 541
    assert compute_loop["PipelineII"] == 1
    assert compute_loop["PipelineDepth"] == 30

    drain_loop = parser.loop_df.loc["poly:drain_loop"]
    assert pd.isna(drain_loop["TripCountMin"])
    assert pd.isna(drain_loop["TripCountMax"])
    assert drain_loop["LatencyMin"] == 7
    assert drain_loop["LatencyMax"] == 7
    assert drain_loop["PipelineII"] == 2
    assert drain_loop["PipelineDepth"] == 4