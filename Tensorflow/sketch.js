function setup() {
    noCanvas(); 

    const values = [];
    for(let i=0; i<15; i++) {
        values[i] = random(0, 100);
    }

    const shape = [5, 3];
    const a = tf.tensor2d(values, shape, 'int32');
    const b = tf.tensor2d(values, shape, 'int32');
    const bb = b.transpose();
    const c = a.matMul(bb);
    a.print();
    bb.print();
    c.print();
}