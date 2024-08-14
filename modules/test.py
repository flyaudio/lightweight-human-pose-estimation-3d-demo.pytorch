'''
dilated convolution example
'''
import torch


def main():
	imput_value = torch.rand(1,1,7,7)
	print("size of feature map = ", imput_value)

	condition_convolution = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1)
	condition_convolution_output = condition_convolution(imput_value)
	print("size of output = ", condition_convolution_output.shape)

	dilated_convolution = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,dilation=2)
	dilated_convolution_output = dilated_convolution(imput_value)
	print("size of output = ", dilated_convolution_output.shape)

	dilated_convolution2 = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,stride=1)
	dilated_convolution_output2 = dilated_convolution2(imput_value)
	print("size of output = ", dilated_convolution_output2.shape)
	print(dilated_convolution_output2)


if __name__ == '__main__':
	main()
