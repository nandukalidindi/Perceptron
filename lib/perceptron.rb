require "perceptron/version"

class Perceptron
  attr_accessor :theta, :alpha, :data, :weights, :bias

  def initialize(theta = 0.2, alpha = 1.0)
    @theta, @alpha = theta, alpha
  end

  def activation(output)
    return 1.0 if output > @theta
    return 0.0 if output >= -@theta && output <= @theta
    return -1.0
  end

  def weights
    @weights ||= (@data.first.length - 1).times.map { |x| 0.0 }
  end

  def bias
    @bias ||= 0.0
  end

  def preprocess_data(file_name = "PerceptronDataF17.txt", delimiter = "\t")
    @data = []
    File.open(file_name, "r") do |f|
      f.each_line do |line|
        @data << line.chomp.split(delimiter).map{ |x| x.to_f }
      end
    end
    data
  end

  def update_weights_and_bias(input)
    new_weights = []
    length = weights.length

    new_weights = (0..length-1).map { |i| weights[i] + @alpha * input[length] * input[i] }
    new_bias = bias + @alpha * input[length]

    @weights, @bias = new_weights, new_bias
  end

  def calculate_output(input)
    yin = (0..weights.length-1).inject(0) { |sum, i| sum + (weights[i] * input[i]) }
    yin + bias
  end

  def process_single_layer
    iterations = 0
    input_count = weights.length
    consecutive_count = 0

    while true
      @data.each do |row|
        iterations += 1
        output = activation(calculate_output(row))
        if output != row[input_count]
          update_weights_and_bias(row)
          consecutive_count = 0
        else
          consecutive_count += 1
          break if consecutive_count >= data.length
        end
      end
      break if consecutive_count >= data.length
    end
    [weights, bias, iterations]
  end
end
