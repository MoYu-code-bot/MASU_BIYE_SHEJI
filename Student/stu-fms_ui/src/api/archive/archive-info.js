import request from '@/utils/request'

const URl = "/archive";

export function list(0, 1) {

  return request.get('/api/archive/list', {
      params: {
        page: page,
        size: size
      }
    }
  ).then(res => {
    console.log(res);

  }).catch(error => {
    console.log("api请求失败", error);
  })
}
