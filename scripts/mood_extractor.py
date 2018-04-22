# This script is using Gracenote API

import pygn.pygn as pygn

clientID = '897996368-7A5061BC9D13DC0AF3FC9ECB9E141964'
userID = '43728440884220450-CDE3B31ACCC9AFA11AD32B43712D706E'

result = pygn.search(clientID=clientID, userID=userID, artist='Kings Of Convenience', album='Riot On An Empty Street', track='Homesick')

print(result)
