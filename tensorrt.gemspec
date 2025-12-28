# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "tensorrt"
  spec.version       = "0.1.0"
  spec.author        = "Pete Matsyburka"
  spec.email         = ["pete@docuseal.com"]

  spec.summary       = "Minimal TensorRT bindings for Ruby using Rice"
  spec.description   = "Minimal Ruby bindings for NVIDIA TensorRT inference using Rice"
  spec.homepage      = "https://github.com/docusealco/tensorrt-rb"
  spec.license       = "Apache-2.0"
  spec.required_ruby_version = ">= 3.0.0"

  spec.files = Dir["lib/**/*.rb", "ext/**/*.{cpp,hpp,rb}", "*.gemspec", "Rakefile", "README.md"]
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/tensorrt_rb/extconf.rb"]

  spec.add_dependency "rice", ">= 4.7"
  spec.add_dependency "ffi", "~> 1.0"

  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rake-compiler", "~> 1.2"
end
