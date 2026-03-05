#include "RunRecord.h"

#include <array>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::string trim_trailing_whitespace(std::string s)
{
	while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
	{
		s.pop_back();
	}
	return s;
}

std::string run_and_capture(const std::string & cmd)
{
	std::array<char, 256> buffer{};
	std::string output;
	FILE * pipe = popen(cmd.c_str(), "r");
	if (pipe == nullptr) { return ""; }

	while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
	{
		output += buffer.data();
	}
	pclose(pipe);
	return trim_trailing_whitespace(output);
}

std::string command_line_from_argv(int argc, char * argv[])
{
	std::ostringstream oss;
	for (int i = 0; i < argc; ++i)
	{
		if (i > 0) { oss << " "; }
		oss << argv[i];
	}
	return oss.str();
}

std::string now_timestamp()
{
	const std::time_t now = std::time(nullptr);
	std::tm tm_buf{};
	localtime_r(&now, &tm_buf);
	std::ostringstream oss;
	oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
	return oss.str();
}

} // namespace

namespace RunRecord {

void write_run_record(int argc, char * argv[], const std::string & file_prefix, const std::string & output_dir)
{
	const std::string timestamp = now_timestamp();
	std::string file_suffix = timestamp;
	for (char & ch : file_suffix)
	{
		if (ch == ' ') { ch = '_'; }
		if (ch == ':') { ch = '-'; }
	}

	std::string record_file = output_dir;
	if (!record_file.empty() && record_file.back() != '/')
	{
		record_file += "/";
	}
	record_file += file_prefix + "_" + file_suffix + ".txt";

	const std::string git_commit = run_and_capture("git rev-parse HEAD 2>/dev/null");
	const std::string git_branch = run_and_capture("git rev-parse --abbrev-ref HEAD 2>/dev/null");

	std::ofstream ofs(record_file);
	if (!ofs.good())
	{
		std::cerr << "Warning: failed to write run record file: " << record_file << std::endl;
		return;
	}

	ofs << "timestamp: " << timestamp << "\n";
	ofs << "command: " << command_line_from_argv(argc, argv) << "\n";
	ofs << "git_commit: " << (git_commit.empty() ? "unknown" : git_commit) << "\n";
	ofs << "git_branch: " << (git_branch.empty() ? "unknown" : git_branch) << "\n";

	std::cout << "Run record written to " << record_file << std::endl;
}

} // namespace RunRecord

