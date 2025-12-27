# frozen_string_literal: true

require "rake/extensiontask"

task default: :compile

Rake::ExtensionTask.new("tensorrt_rb") do |ext|
  ext.lib_dir = "lib/tensorrt_rb"
end

desc "Build and install the gem"
task :install => :compile do
  sh "gem build tensorrt-rb.gemspec && gem install tensorrt-rb-*.gem"
end

desc "Clean build artifacts"
task :clean do
  rm_rf "tmp"
  rm_rf "lib/tensorrt_rb/*.bundle"
  rm_rf "lib/tensorrt_rb/*.so"
  rm_f Dir.glob("*.gem")
end
