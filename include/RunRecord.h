#pragma once

#include <string>

namespace RunRecord {

// Write a run record file with timestamp, command line, git commit and branch.
// Example output file name: run_record_2026-03-05_15-06-06.txt
void write_run_record(
	int argc,
	char * argv[],
	const std::string & file_prefix = "run_record",
	const std::string & output_dir = "."
);

} // namespace RunRecord

