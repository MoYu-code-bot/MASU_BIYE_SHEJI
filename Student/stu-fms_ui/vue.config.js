module.exports = {
  lintOnSave: false, // 关闭lint检查
  // 解决跨域
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:8082',
        secure: false, // 使用的是http协议则设置为false，https协议则设置为true
        changOrigin: true, // 开启代理
        ws: true,
        pathRewrite: {
          '^/api': ''
        },
        host: '0.0.0.0'
      }
    }
  }
}
