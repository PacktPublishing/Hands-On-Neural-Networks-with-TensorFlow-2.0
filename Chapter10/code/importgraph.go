package main

import (
	"fmt"
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	model := tg.LoadModel("/tmp/pow/1", []string{"serve"}, nil)

	x, _ := tf.NewTensor(float32(2.0))
	y, _ := tf.NewTensor(float32(5.0))

	results := model.Exec([]tf.Output{
		model.Op("PartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("serving_default_x", 0): x,
		model.Op("serving_default_y", 0): y,
	})

	predictions := results[0].Value().(float32)
	fmt.Println(predictions)

}
