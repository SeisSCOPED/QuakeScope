#!/bin/zsh
# Does the EarthScope role allow GetObject on the restricted access point,
# or only ListBucket?  Prints one verdict line per network.
#
#   ./scripts/check_earthscope_getobject.sh            # default networks
#   ./scripts/check_earthscope_getobject.sh CC NN UO   # specific ones
#
# 403 = object exists, you are NOT entitled to read it  -> ask EarthScope
# 200 = readable                                        -> you have access
# 404 = no data that day (harmless; try another network)

AP=${EARTHSCOPE_S3_ACCESS_POINT:-earthscope-mseed-v2-4fdodyzpsz8u8uyi3pa9qsw9oid1suse2a-s3alias}
ROLE=${EARTHSCOPE_ROLE:-s3-miniseed-v2}
if (( $# )); then NETS=($@); else NETS=(CC NN NP UO); fi
DAY=${DAY:-2019/187}

eval $(pixi run -e cloud python -c "
from earthscope_sdk import EarthScopeClient
with EarthScopeClient() as c: r=c.user.get_aws_credentials(role='$ROLE')
print(f'export AWS_ACCESS_KEY_ID={r.aws_access_key_id}')
print(f'export AWS_SECRET_ACCESS_KEY={r.aws_secret_access_key}')
print(f'export AWS_SESSION_TOKEN={r.aws_session_token}')
print('export AWS_DEFAULT_REGION=us-east-2')") || {
  echo "Could not get credentials for role $ROLE - is ES_OAUTH2__REFRESH_TOKEN set?"; exit 2; }

echo "access point : $AP"
echo "role         : $ROLE"
printf '%-6s %-6s %s\n' NET LIST 'GET (head-object)'
rc=0
for NET in $NETS; do
  KEY=$(aws s3api list-objects-v2 --bucket $AP --prefix "miniseed/$NET/$DAY/" \
        --max-keys 1 --query 'Contents[0].Key' --output text 2>/dev/null)
  if [[ -z "$KEY" || "$KEY" == "None" ]]; then
    printf '%-6s %-6s %s\n' $NET "DENY" "cannot even list - no data that day, or no ListBucket"; rc=2; continue
  fi
  ERR=$(aws s3api head-object --bucket $AP --key "$KEY" 2>&1 >/dev/null)
  if [[ -z "$ERR" ]]; then
    printf '%-6s %-6s %s\n' $NET "ok" "200 READABLE  ($KEY)"
  elif [[ "$ERR" == *"(403)"* ]]; then
    printf '%-6s %-6s %s\n' $NET "ok" "403 FORBIDDEN - object exists, not entitled  ($KEY)"; rc=1
  else
    printf '%-6s %-6s %s\n' $NET "ok" "$(echo $ERR | grep -o 'An error.*' | head -1)"; rc=2
  fi
done
echo
[[ $rc -eq 1 ]] && echo "VERDICT: listing works, reading does not -> EarthScope entitlement gap."
[[ $rc -eq 0 ]] && echo "VERDICT: GetObject granted - the restricted tier is unblocked."
exit $rc
